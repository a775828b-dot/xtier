// Cold experts on NVMe (xtier/cache/cold.py): expert records in the CPU engine's shard layout, kept in one file per
// drive and read with O_DIRECT straight into RAM slots (xtier_slot_pieces). Writes go through an aligned bounce
// buffer, also O_DIRECT: nothing passes the page cache, so what is measured is the drives, and the store never takes
// host memory beyond its buffers. Offsets and piece sizes are multiples of 512 bytes (the drives' logical block).
// Reads are synchronous (xtier_cold_read: a forward waits for them, ahead of the others) or asynchronous
// (xtier_cold_submit: a ticket, waited for by groups of jobs, e.g. the experts of a prefill's staging).
#include "xtier_engine.h"
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
thread_local std::string cold_error;
using Clock=std::chrono::steady_clock;

// One read: its jobs by group (counters under the store's done_lock).
struct Ticket {
    std::vector<int> pending;   // per group: jobs not done
    int left=0,failed=0;
    long long bytes=0;
    Clock::time_point start,end;
};

struct Job {
    long long offset;
    std::vector<iovec> pieces;
    Ticket *ticket;
    int group;
};

struct File {
    int fd=-1;
    std::mutex lock;
    std::condition_variable wake;
    std::deque<Job> urgent,jobs;   // urgent: the reads a forward waits for (xtier_cold_read)
    std::vector<std::thread> readers;
    bool stop=false;
};

struct Store {
    std::vector<File*> files;
    std::mutex done_lock;
    std::condition_variable done;
    long long bytes_read=0,bytes_written=0,reads=0;
    double read_seconds=0;
};

// Whole-request preadv: short reads continue from where they stopped.
bool read_all(int fd,long long offset,std::vector<iovec> v) {
    size_t i=0;
    while(i<v.size()) {
        ssize_t got;
        do {got=preadv(fd,v.data()+i,int(v.size()-i),offset);} while(got<0&&errno==EINTR);
        if(got<=0)return false;
        offset+=got;
        while(i<v.size()&&size_t(got)>=v[i].iov_len){got-=ssize_t(v[i].iov_len);i++;}
        if(i<v.size()){v[i].iov_base=static_cast<uint8_t*>(v[i].iov_base)+got;v[i].iov_len-=size_t(got);}
    }
    return true;
}

void reader(Store *store,File *file) {
    for(;;) {
        Job job;
        {
            std::unique_lock<std::mutex> g(file->lock);
            file->wake.wait(g,[&]{return file->stop||!file->urgent.empty()||!file->jobs.empty();});
            if(file->stop&&file->urgent.empty()&&file->jobs.empty())return;
            auto &q=file->urgent.empty()?file->jobs:file->urgent;
            job=std::move(q.front());q.pop_front();
        }
        bool ok=read_all(file->fd,job.offset,job.pieces);
        std::lock_guard<std::mutex> g(store->done_lock);
        Ticket *t=job.ticket;
        if(!ok)t->failed++;
        t->pending[size_t(job.group)]--;
        if(--t->left==0)t->end=Clock::now();
        store->done.notify_all();
    }
}

// Checks every job (before any is queued: queued jobs point at the ticket's counters), then queues them.
int queue(Store *store,Ticket *t,int jobs,const int *files,const long long *offsets,void *const *dst,const size_t *sizes,
          int n,const int *groups,bool urgent) {
    for(int j=0;j<jobs;j++) {
        if(files[j]<0||files[j]>=int(store->files.size())||offsets[j]<0||offsets[j]%512){cold_error="Invalid read job";return -1;}
        if(groups&&(groups[j]<0||groups[j]>=int(t->pending.size()))){cold_error="Invalid read group";return -1;}
    }
    t->left=jobs;t->start=t->end=Clock::now();
    for(int j=0;j<jobs;j++)t->pending[groups?size_t(groups[j]):0]++;
    for(int j=0;j<jobs;j++) {
        Job job{offsets[j],{},t,groups?groups[j]:0};
        for(int q=0;q<n;q++){job.pieces.push_back({dst[size_t(j)*n+q],sizes[size_t(j)*n+q]});t->bytes+=sizes[size_t(j)*n+q];}
        File *file=store->files[files[j]];
        {std::lock_guard<std::mutex> g(file->lock);(urgent?file->urgent:file->jobs).push_back(std::move(job));}
        file->wake.notify_one();
    }
    return 0;
}

// The ticket's jobs are done (done_lock held): counted in the store's statistics.
int account(Store *store,Ticket *t,int jobs) {
    if(t->failed){cold_error="Read failed";return -1;}
    store->bytes_read+=t->bytes;store->reads+=jobs;
    store->read_seconds+=std::chrono::duration<double>(t->end-t->start).count();
    return 0;
}

struct Async {   // an xtier_cold_submit ticket
    Ticket ticket;
    int jobs=0;
};
}

extern "C" {
const char *xtier_cold_error(){return cold_error.c_str();}

void *xtier_cold_open(const char *const *paths,int count,int threads,int writable) {
    if(!paths||count<1||threads<1||threads>64){cold_error="Invalid store arguments";return nullptr;}
    auto *store=new Store();
    for(int i=0;i<count;i++) {
        auto *file=new File();
        file->fd=open(paths[i],(writable?O_RDWR|O_CREAT:O_RDONLY)|O_DIRECT|O_CLOEXEC,0644);
        store->files.push_back(file);
        if(file->fd<0) {
            cold_error=std::string("Cannot open ")+paths[i]+": "+std::strerror(errno);
            xtier_cold_close(store);return nullptr;
        }
        for(int t=0;t<threads;t++)file->readers.emplace_back(reader,store,file);
    }
    return store;
}

int xtier_cold_write(void *handle,int file,long long offset,const void *const *pieces,const size_t *sizes,int n) {
    auto *store=static_cast<Store*>(handle);
    if(!store||file<0||file>=int(store->files.size())||offset<0||offset%512||n<1){cold_error="Invalid write";return -1;}
    size_t total=0;for(int j=0;j<n;j++)total+=sizes[j];
    if(total%512){cold_error="Record size must be a multiple of 512";return -1;}
    void *bounce=nullptr;
    if(posix_memalign(&bounce,4096,total)){cold_error="Out of memory";return -1;}
    size_t at=0;for(int j=0;j<n;j++){std::memcpy(static_cast<uint8_t*>(bounce)+at,pieces[j],sizes[j]);at+=sizes[j];}
    size_t done=0;int rc=0;
    while(done<total) {
        ssize_t put=pwrite(store->files[file]->fd,static_cast<uint8_t*>(bounce)+done,total-done,offset+done);
        if(put<0&&errno==EINTR)continue;
        if(put<=0){cold_error=std::string("Write failed: ")+std::strerror(errno);rc=-1;break;}
        done+=size_t(put);
    }
    std::free(bounce);
    if(rc==0){std::lock_guard<std::mutex> g(store->done_lock);store->bytes_written+=total;}
    return rc;
}

int xtier_cold_read(void *handle,int jobs,const int *files,const long long *offsets,void *const *dst,const size_t *sizes,
                    int n) {
    auto *store=static_cast<Store*>(handle);
    if(!store||jobs<0||n<1||(jobs&&(!files||!offsets||!dst||!sizes))){cold_error="Invalid read";return -1;}
    if(!jobs)return 0;
    Ticket t;t.pending.assign(1,0);
    if(queue(store,&t,jobs,files,offsets,dst,sizes,n,nullptr,true))return -1;
    std::unique_lock<std::mutex> g(store->done_lock);
    store->done.wait(g,[&]{return t.left==0;});
    return account(store,&t,jobs);
}

void *xtier_cold_submit(void *handle,int jobs,const int *files,const long long *offsets,void *const *dst,
                        const size_t *sizes,int n,const int *groups,int group_count) {
    auto *store=static_cast<Store*>(handle);
    if(!store||jobs<0||n<1||group_count<1||(jobs&&(!files||!offsets||!dst||!sizes||!groups))){cold_error="Invalid read";return nullptr;}
    auto *a=new Async();
    a->ticket.pending.assign(size_t(group_count),0);a->jobs=jobs;
    if(jobs&&queue(store,&a->ticket,jobs,files,offsets,dst,sizes,n,groups,false)){delete a;return nullptr;}
    return a;
}

int xtier_cold_wait(void *handle,void *ticket,int g0,int g1) {
    auto *store=static_cast<Store*>(handle);auto *a=static_cast<Async*>(ticket);
    if(!store||!a||g0<0||g1>int(a->ticket.pending.size())){cold_error="Invalid wait";return -1;}
    std::unique_lock<std::mutex> g(store->done_lock);
    store->done.wait(g,[&]{for(int i=g0;i<g1;i++)if(a->ticket.pending[size_t(i)])return false;return true;});
    if(a->ticket.failed){cold_error="Read failed";return -1;}
    return 0;
}

int xtier_cold_finish(void *handle,void *ticket) {
    auto *store=static_cast<Store*>(handle);auto *a=static_cast<Async*>(ticket);
    if(!store||!a){cold_error="Invalid finish";return -1;}
    int rc;
    {
        std::unique_lock<std::mutex> g(store->done_lock);
        store->done.wait(g,[&]{return a->ticket.left==0;});
        rc=a->jobs?account(store,&a->ticket,a->jobs):0;
    }
    delete a;
    return rc;
}

void xtier_cold_stats(void *handle,double *out) {
    auto *store=static_cast<Store*>(handle);
    if(!store||!out)return;
    std::lock_guard<std::mutex> g(store->done_lock);
    out[0]=double(store->bytes_read);out[1]=double(store->bytes_written);out[2]=double(store->reads);out[3]=store->read_seconds;
}

void xtier_cold_close(void *handle) {
    auto *store=static_cast<Store*>(handle);
    if(!store)return;
    for(File *file:store->files) {
        {std::lock_guard<std::mutex> g(file->lock);file->stop=true;}
        file->wake.notify_all();
        for(auto &t:file->readers)t.join();
        if(file->fd>=0)close(file->fd);
        delete file;
    }
    delete store;
}
}
