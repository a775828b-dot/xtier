// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments
// renamed only, so the kernels compile to the same code.
// Standalone mailbox bridge (production path of the WebUI launcher since 2026-09-27).
// One shared CPU dispatcher polls mapped host mailboxes. Stream mem-ops replace
// CUDA host callbacks; all graph dependencies remain explicit and stream ordered.
// Each slot must be used serially; destroy its graphs before closing the plan.
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include "xtier_engine.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int MAX_SLOT_M=8; // decode tokens per slot (DSpark verify uses 6)
thread_local std::string error;
// XTIER_ROUTE_TRACE=<file> (research, default off): the dispatcher thread appends one record per CPU
// decode call: TraceHead (realtime ns at start, wall/gate/down ns, layer = bridge creation order, m, k,
// expert groups or -1, active tokens) followed by the m*k int32 route ids as the GPU sent them.
struct TraceHead {uint64_t t_ns;uint32_t wall_ns,gate_ns,down_ns;uint16_t layer;uint8_t m,k;int16_t groups;uint8_t active,pad;uint32_t seq;};
static_assert(sizeof(TraceHead)==32,"trace record header");
std::FILE *trace_file() {
    static std::FILE *f=[]()->std::FILE* {
        const char *p=std::getenv("XTIER_ROUTE_TRACE");
        if(!p||!*p)return nullptr;
        std::FILE *t=std::fopen(p,"ab");
        if(!t){std::fprintf(stderr,"[xtier] route trace %s: cannot open\n",p);return nullptr;}
        std::setvbuf(t,nullptr,_IOFBF,1<<22);
        std::fprintf(stderr,"[xtier] route trace -> %s\n",p);
        return t;
    }();
    return f;
}
std::atomic<int> bridge_count{0};
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
void driver(CUresult rc) {
    if(rc!=CUDA_SUCCESS){const char *s=nullptr;cuGetErrorString(rc,&s);throw std::runtime_error(s?s:"CUDA driver error");}
}
struct Bridge;
struct Slot;
class Dispatcher {
    std::mutex mutex;
    std::vector<Slot*> slots;
    std::atomic<bool> quit{false};
    std::thread worker,watchdog;
    std::atomic<int64_t> busy_since{0}; // steady-clock ns while a CPU call runs, else 0
public:
    Dispatcher();
    ~Dispatcher();
    void add(Slot *s);
    void remove(Slot *s);
};
std::shared_ptr<Dispatcher> get_dispatcher();
struct Slot {
    Bridge *owner;
    cudaStream_t stream;
    int m,k,H; // H: hidden size of the owning plan
    bool shadow;
    void *storage=nullptr;
    uint32_t *mailbox=nullptr;
    int64_t pending_since=0; // watchdog only: first time the request was seen unserved
    CUdeviceptr device_mailbox=0;
    uint16_t *x=nullptr;
    int32_t *ids=nullptr;
    float *routing=nullptr,*out=nullptr,*reference=nullptr;
    int32_t *orig=nullptr; // expert pool: the routes as routed (ids holds -1 for the GPU-resident ones)
    char *device_storage=nullptr; // device-visible alias of storage for kernel-side staging
    std::vector<uint16_t> cx; std::vector<int32_t> cids; std::vector<float> croute,cout; // compaction scratch
    Slot(Bridge*b,cudaStream_t st,int rows,int topk,bool sh,int hidden):owner(b),stream(st),m(rows),k(topk),H(hidden),shadow(sh) {
        auto align=[](size_t n){return (n+255)&~size_t(255);};
        size_t xbytes=align(size_t(m)*H*2),ibytes=align(m*k*4),rbytes=align(m*k*4),obytes=align(size_t(m)*H*4);
        check(cudaHostAlloc(&storage,xbytes+ibytes+rbytes+2*obytes+ibytes,cudaHostAllocMapped));
        auto *p=static_cast<char*>(storage);
        x=reinterpret_cast<uint16_t*>(p);p+=xbytes;ids=reinterpret_cast<int32_t*>(p);p+=ibytes;
        routing=reinterpret_cast<float*>(p);p+=rbytes;out=reinterpret_cast<float*>(p);p+=obytes;
        reference=reinterpret_cast<float*>(p);p+=obytes;
        orig=reinterpret_cast<int32_t*>(p);std::fill(orig,orig+m*k,-1);
        try {
            check(cudaHostAlloc(reinterpret_cast<void**>(&mailbox),256,cudaHostAllocMapped));
            __atomic_store_n(mailbox,0,__ATOMIC_RELEASE);
            void *device=nullptr;check(cudaHostGetDevicePointer(&device,mailbox,0));
            device_mailbox=reinterpret_cast<CUdeviceptr>(device);
            check(cudaHostGetDevicePointer(&device,storage,0));device_storage=static_cast<char*>(device);
        } catch(...) {if(mailbox)cudaFreeHost(mailbox);cudaFreeHost(storage);throw;}
    }
    ~Slot(){if(mailbox)cudaFreeHost(mailbox);if(storage)cudaFreeHost(storage);}
};
struct Bridge {
    void *plan;
    int H=0;
    int index=bridge_count.fetch_add(1); // creation order = CPU MoE layer order (route trace)
    std::shared_ptr<Dispatcher> dispatcher;
    std::vector<std::unique_ptr<Slot>> slots;
    std::mutex slot_mutex,stats_mutex;
    std::atomic<int> failed{0};
    // calls, elements, sum_delta2, sum_ref2, max_abs, max_call_rel_l2, nonfinite
    std::array<double,7> stats{};
    // Per M=1..MAX_SLOT_M: calls, execute wall us, gate+barrier us, down+barrier us.
    std::array<double,4*MAX_SLOT_M> timing{};
    // XTier cache expert pool (xtier_mailbox_pool_enable): the packing kernel also stores the original routes;
    // calls, distinct experts, distinct experts served by the GPU pool (written by the dispatcher).
    std::atomic<int> pool{0};
    std::array<double,3> pool_stats{};
    int pool_layer=-1;   // index in the pool manager (adaptive residency), -1 = not attached
    explicit Bridge(void *p):plan(xtier_retain(p)) {
        if(!plan)throw std::runtime_error(xtier_last_error());
        int shape[4];
        if(xtier_shape(plan,shape)<0){xtier_destroy(plan);throw std::runtime_error(xtier_last_error());}
        H=shape[0];
        try {dispatcher=get_dispatcher();}catch(...){xtier_destroy(plan);throw;}
    }
    ~Bridge(){for(auto &s:slots)dispatcher->remove(s.get());slots.clear();dispatcher.reset();xtier_destroy(plan);}
    Slot* find(cudaStream_t stream,int m,int k,bool shadow) {
        for(auto &s:slots)if(s->stream==stream&&s->m==m&&s->k==k&&s->shadow==shadow)return s.get();
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(stream,&status));
        if(status!=cudaStreamCaptureStatusNone)throw std::runtime_error("Prepare this stream/shape before graph capture");
        slots.push_back(std::make_unique<Slot>(this,stream,m,k,shadow,H));
        try {dispatcher->add(slots.back().get());}
        catch(...) {slots.pop_back();throw;}
        return slots.back().get();
    }
};
// ---- XTier cache expert pool, adaptive residency (xtier_mailbox_pool_attach / _start; XTIER_CACHE_ADAPT) ---------
// The dispatcher counts every pool layer's routes (decayed counts). Every `every` decode windows the policy
// thread swaps the most-routed non-resident experts in for the least-routed residents (gain > min_gain, at
// most budget*every per round over all layers, best gain first), like Strata's adaptive tier
// (github.com/Niko1221/Strata, MIT; this is an independent implementation).
// A swap is a command in a mapped ring: the GPU's commit kernel (first node of every decode window, see
// pool.py) removes the old expert from the device residency table and acknowledges (applied); the policy
// thread waits on the host for that acknowledgement, then enqueues the upload of the new expert from the
// pinned NUMA shards on the copy stream, followed by a write of `landed`; the next commit publishes the new
// expert. No decode kernel ever reads a slot that is being written, and both halves of a call see one table.
// Nothing on the copy stream ever waits for the decode stream: a device-wide synchronize (sglang does them)
// only waits for copies already in flight.
struct PoolCmd {int32_t layer,in,out,slot;};
constexpr uint32_t POOL_RING=4096;
struct PoolLayer {
    Bridge *bridge=nullptr;
    int layer=0,S=0,E=0,H=0,I=0,group=16;
    char *arena[4]{};                 // device: w13, w2, s13, s2 slots (xtier shard layout)
    int nodes=0;                      // xtier worker nodes (each holds 1/nodes of every expert's rows)
    const char *shard[16][6]{};       // host (pinned): per worker node, gate, up, gate scales, up scales, down, down scales
    std::vector<int> resident;        // slot -> expert (-1 empty), as decided (copies may be in flight)
    std::vector<int> slot_of;         // expert -> slot, as decided
    std::vector<uint32_t> busy;       // slot -> seq of its in-flight copy (0 = none)
    std::vector<float> usage;         // decayed route counts
};
class PoolManager {
public:
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::unique_ptr<PoolLayer>> layers;   // by layer index
    PoolCmd *ring=nullptr;uint32_t *seqs=nullptr;     // mapped host: ring, [posted, applied, landed] (64 B apart)
    CUdeviceptr dev_ring=0,dev_seqs=0;
    cudaStream_t copy=nullptr;
    int every=4,budget=16;float decay=0.7f,min_gain=1.5f;
    bool running=false,quit=false;
    long windows=0,round_at=0,swaps=0,rounds=0;
    // Prompt warm-up (XTIER_CACHE_PROMPT, pool.py record_prompt): a prefill's route counts, scaled to warm_windows
    // decode windows' worth, blend into the counts (the old ones x warm_keep), and the next warm_rounds rounds run
    // at once (from the next decode window) with up to warm_budget swaps each: the decode that follows finds the
    // prompt's hot experts instead of the previous request's (the idea of Strata 0.2.4's cache_settle).
    float warm_keep=0.7f,warm_windows=10.f;int warm_rounds=4,warm_budget=256;
    std::atomic<int> warm_left{0};
    double route_avg=0.;   // routes per pool layer call of decode: the size of a window's counts
    long prompts=0;
    uint32_t next_seq=0;
    std::thread worker;
    uint32_t *posted(){return seqs;}uint32_t *applied(){return seqs+16;}uint32_t *landed(){return seqs+32;}
    void ensure_ring() {
        if(ring)return;
        check(cudaHostAlloc(reinterpret_cast<void**>(&ring),sizeof(PoolCmd)*POOL_RING,cudaHostAllocMapped));
        check(cudaHostAlloc(reinterpret_cast<void**>(&seqs),256,cudaHostAllocMapped));
        std::memset(ring,0,sizeof(PoolCmd)*POOL_RING);std::memset(seqs,0,256);
        void *d=nullptr;
        check(cudaHostGetDevicePointer(&d,ring,0));dev_ring=reinterpret_cast<CUdeviceptr>(d);
        check(cudaHostGetDevicePointer(&d,seqs,0));dev_seqs=reinterpret_cast<CUdeviceptr>(d);
    }
    // dispatcher thread, every pool layer call: count the routes as routed
    void observe(int layer,const int32_t *orig,int n) {
        std::unique_lock<std::mutex> lock(mu,std::try_to_lock);
        if(!lock.owns_lock())return;   // the policy holds the counts for a moment: drop this call's counts
        if(layer<0||size_t(layer)>=layers.size()||!layers[layer])return;
        auto &u=layers[layer]->usage;
        for(int i=0;i<n;i++)if(orig[i]>=0&&orig[i]<int(u.size()))u[orig[i]]+=1.f;
        route_avg=route_avg>0.?0.99*route_avg+0.01*n:double(n);
        if(layer==0&&running&&(++windows-round_at>=every||warm_left.load(std::memory_order_relaxed)>0)) {
            round_at=windows;lock.unlock();cv.notify_one();
        }
    }
    // a prefill's route counts per pool layer (row l of `stride` floats), from a CUDA host function after their copy
    void prompt(const float *counts,int n_layers,int stride) {
        std::lock_guard<std::mutex> lock(mu);
        const double window=route_avg>0.?route_avg:36.;
        bool any=false;
        for(int l=0;l<n_layers&&size_t(l)<layers.size();l++) {
            auto *L=layers[l].get();if(!L)continue;
            const float *row=counts+size_t(l)*stride;
            const int E=std::min(L->E,stride);
            double total=0.;for(int e=0;e<E;e++)total+=row[e];
            if(total<=0.)continue;
            const float scale=float(warm_windows*window/total);
            for(int e=0;e<E;e++)L->usage[e]=L->usage[e]*warm_keep+row[e]*scale;
            any=true;
        }
        if(any){warm_left.store(warm_rounds,std::memory_order_relaxed);prompts++;}
    }
    void upload(PoolLayer &L,int expert,int slot) {
        const size_t H=L.H,I=L.I,IN=I/L.nodes,HN=H/L.nodes;
        // (arena, first row, rows per node, total rows, bytes per row) of each shard tensor, as prefill._shard_layout
        const size_t lay[6][5]={{0,0,IN,2*I,H/2},{0,I,IN,2*I,H/2},{2,0,IN,2*I,H/L.group},{2,I,IN,2*I,H/L.group},
                                {1,0,HN,H,I/2},{3,0,HN,H,I/L.group}};
        for(int node=0;node<L.nodes;node++)for(int j=0;j<6;j++) {
            const size_t *q=lay[j];
            char *dst=L.arena[q[0]]+size_t(slot)*q[3]*q[4]+(q[1]+node*q[2])*q[4];
            const char *src=L.shard[node][j]+size_t(expert)*q[2]*q[4];
            check(cudaMemcpyAsync(dst,src,q[2]*q[4],cudaMemcpyHostToDevice,copy));
        }
    }
    struct Copy {PoolLayer *L;int in,slot;uint32_t seq;};
    uint32_t post(PoolLayer &L,int in,int slot) {
        int out=L.resident[slot];
        uint32_t seq=++next_seq;
        ring[seq%POOL_RING]=PoolCmd{L.layer,in,out,slot};
        __atomic_store_n(posted(),seq,__ATOMIC_RELEASE);
        if(out>=0)L.slot_of[out]=-1;
        L.resident[slot]=in;L.slot_of[in]=slot;L.busy[slot]=seq;
        return seq;
    }
    // After the GPU acknowledged the evictions (next decode window): upload, then mark each command landed.
    void copy_after_applied(const std::vector<Copy> &todo) {
        if(todo.empty())return;
        const uint32_t last=todo.back().seq;
        while(int32_t(__atomic_load_n(applied(),__ATOMIC_ACQUIRE)-last)<0) {
            if(quit)return;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        for(auto &c:todo) {
            upload(*c.L,c.in,c.slot);
            driver(cuStreamWriteValue32(reinterpret_cast<CUstream>(copy),dev_seqs+32*4,c.seq,0));
        }
    }
    void round() {
        struct Swap {float gain;int layer,in,slot;};
        std::vector<Swap> swaps_round;
        std::vector<std::vector<float>> u;
        {std::lock_guard<std::mutex> lock(mu);for(auto &L:layers)u.push_back(L?L->usage:std::vector<float>());}
        const uint32_t done=__atomic_load_n(landed(),__ATOMIC_ACQUIRE);
        std::vector<std::pair<float,int>> cand,vict;
        for(size_t l=0;l<layers.size();l++) {
            auto *L=layers[l].get();if(!L)continue;
            cand.clear();vict.clear();
            for(int e=0;e<L->E;e++)if(L->slot_of[e]<0&&u[l][e]>=2.f)cand.emplace_back(u[l][e],e);
            for(int s=0;s<L->S;s++) {
                if(L->busy[s]&&L->busy[s]>done)continue;   // its copy has not landed yet
                L->busy[s]=0;
                vict.emplace_back(L->resident[s]<0?-1.f:u[l][L->resident[s]],s);
            }
            if(cand.empty()||vict.empty())continue;
            size_t nc=std::min(cand.size(),vict.size());
            std::partial_sort(cand.begin(),cand.begin()+nc,cand.end(),[](auto&a,auto&b){return a.first>b.first;});
            std::partial_sort(vict.begin(),vict.begin()+nc,vict.end(),[](auto&a,auto&b){return a.first<b.first;});
            for(size_t i=0;i<nc;i++) {
                if(cand[i].first<vict[i].first+min_gain)break;
                swaps_round.push_back({cand[i].first-vict[i].first,int(l),cand[i].second,vict[i].second});
            }
        }
        std::sort(swaps_round.begin(),swaps_round.end(),[](const Swap&a,const Swap&b){return a.gain>b.gain;});
        const bool warm=warm_left.load(std::memory_order_relaxed)>0;
        if(warm)warm_left.fetch_sub(1,std::memory_order_relaxed);
        size_t limit=size_t(std::max(0,warm?warm_budget:budget*every));
        // never more commands in flight than the ring holds
        uint32_t applied_now=__atomic_load_n(applied(),__ATOMIC_ACQUIRE);
        size_t room=POOL_RING/2>next_seq-applied_now?POOL_RING/2-(next_seq-applied_now):0;
        limit=std::min(limit,room);
        if(swaps_round.size()>limit)swaps_round.resize(limit);
        std::vector<Copy> todo;
        for(auto &s:swaps_round){auto *L=layers[s.layer].get();todo.push_back({L,s.in,s.slot,post(*L,s.in,s.slot)});}
        {std::lock_guard<std::mutex> lock(mu);for(auto &L:layers)if(L)for(auto &v:L->usage)v*=decay;}
        swaps+=long(swaps_round.size());rounds++;
        copy_after_applied(todo);
    }
    void start() {
        if(running)return;
        int lo=0,hi=0;check(cudaDeviceGetStreamPriorityRange(&lo,&hi));
        check(cudaStreamCreateWithPriority(&copy,cudaStreamNonBlocking,lo));   // lowest priority
        running=true;
        // The policy thread uploads with the CUDA runtime and driver: give it this caller's device
        // (one process per tensor-parallel rank, each on its own GPU; a new thread starts on device 0).
        int device=0;check(cudaGetDevice(&device));
        worker=std::thread([this,device]{
            if(cudaSetDevice(device)!=cudaSuccess){std::fprintf(stderr,"[xtier] XTier cache policy stopped: cannot select device %d\n",device);return;}
            for(;;) {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock);
                if(quit)return;
                lock.unlock();
                try{round();}
                catch(const std::exception &e){std::fprintf(stderr,"[xtier] XTier cache policy stopped: %s\n",e.what());return;}
            }
        });
        worker.detach();   // runs until the process exits (copies may wait on a stopped decode stream)
    }
};
PoolManager &pool_manager(){static PoolManager *m=new PoolManager();return *m;}

void CUDART_CB execute(void *p) noexcept {
    auto *s=static_cast<Slot*>(p);auto *b=s->owner;
    // Test hook for the watchdog only; unset in normal use.
    static const long stall_us=[]{const char *v=std::getenv("XTIER_DEBUG_STALL_US");return v?std::stol(v):0L;}();
    if(stall_us)std::this_thread::sleep_for(std::chrono::microseconds(stall_us));
    timespec real{};if(trace_file())clock_gettime(CLOCK_REALTIME,&real);
    auto begin=std::chrono::steady_clock::now();
    // Tokens whose route ids are -1 were skipped on the GPU (XTIER_ACTIVE_FLAGS: e.g. a truncated
    // speculative verify): compute only the others, in order, and return zero rows for them.
    const int m=s->m,k=s->k,H=s->H;
    // A token is computed when any of its routes is >= 0: with the expert pool, routes served by the
    // GPU are -1 too, and a token whose routes are all on the GPU gets a zero CPU row.
    auto live=[&](int t){for(int q=0;q<k;q++)if(s->ids[t*k+q]>=0)return true;return false;};
    int active=0;bool holes=false;
    for(int t=0;t<m;t++)if(live(t)){holes|=active!=t;active++;}
    int rc=0;
    if(active==m)rc=xtier_forward(b->plan,m,k,s->x,s->ids,s->routing,s->out);
    else if(!holes) {
        if(active)rc=xtier_forward(b->plan,active,k,s->x,s->ids,s->routing,s->out);
        std::fill(s->out+size_t(active)*H,s->out+size_t(m)*H,0.f);
    } else {
        s->cx.resize(size_t(m)*H);s->cids.resize(m*k);s->croute.resize(m*k);s->cout.resize(size_t(m)*H);
        int n=0;
        for(int t=0;t<m;t++)if(live(t)) {
            std::copy(s->x+size_t(t)*H,s->x+size_t(t+1)*H,s->cx.data()+size_t(n)*H);
            std::copy(s->ids+t*k,s->ids+(t+1)*k,s->cids.data()+n*k);
            std::copy(s->routing+t*k,s->routing+(t+1)*k,s->croute.data()+n*k);n++;
        }
        rc=xtier_forward(b->plan,n,k,s->cx.data(),s->cids.data(),s->croute.data(),s->cout.data());
        n=0;
        for(int t=0;t<m;t++) {
            float *row=s->out+size_t(t)*H;
            if(live(t)){std::copy(s->cout.data()+size_t(n)*H,s->cout.data()+size_t(n+1)*H,row);n++;}
            else std::fill(row,row+H,0.f);
        }
    }
    double phases[2]{};
    if(rc>=0)xtier_last_timings(b->plan,phases);
    double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
    if(rc<0) {
        // Graph replays never re-check `failed`; make the NaN output visible in logs.
        static std::atomic<int> reported{0};
        if(reported.fetch_add(1)<16)
            std::fprintf(stderr,"[xtier] CPU forward failed (m=%d k=%d): %s; output set to NaN\n",
                         s->m,s->k,xtier_last_error());
        b->failed.store(1,std::memory_order_release);
        std::fill(s->out,s->out+s->m*s->H,std::numeric_limits<float>::quiet_NaN());
    }
    if(s->shadow) {
        double d2=0,r2=0,mx=0,bad=0;
        for(int i=0;i<s->m*s->H;i++) {
            double a=s->out[i],r=s->reference[i],d=a-r;
            if(!std::isfinite(a)||!std::isfinite(r)){bad++;continue;}
            d2+=d*d;r2+=r*r;mx=std::max(mx,std::abs(d));
        }
        std::lock_guard<std::mutex> lock(b->stats_mutex);
        auto &t=b->stats;t[0]++;t[1]+=s->m*s->H;t[2]+=d2;t[3]+=r2;
        t[4]=std::max(t[4],mx);t[5]=std::max(t[5],std::sqrt(d2/std::max(r2,1e-30)));t[6]+=bad;
    } else {
        std::lock_guard<std::mutex> lock(b->stats_mutex);b->stats[0]++;b->stats[1]+=s->m*s->H;
    }
    if(rc>=0&&active>0) {
        std::lock_guard<std::mutex> lock(b->stats_mutex);
        double *t=b->timing.data()+4*(active-1);t[0]++;t[1]+=wall;t[2]+=phases[0];t[3]+=phases[1];
    }
    const bool pool=b->pool.load(std::memory_order_relaxed)!=0;
    if(pool) {
        // distinct experts of the call (original routes; skipped tokens are -1) and how many the GPU served
        int seen[MAX_SLOT_M*10],n=0,hit=0;
        for(int i=0;i<m*k;i++) {
            int e=s->orig[i];if(e<0)continue;
            bool dup=false;for(int j=0;j<n;j++)if(seen[j]==e){dup=true;break;}
            if(dup)continue;
            seen[n++]=e;if(s->ids[i]<0)hit++;
        }
        {std::lock_guard<std::mutex> lock(b->stats_mutex);auto &ps=b->pool_stats;ps[0]++;ps[1]+=n;ps[2]+=hit;}
        if(b->pool_layer>=0)pool_manager().observe(b->pool_layer,s->orig,m*k);
        static const long every=[]{const char *v=std::getenv("XTIER_CACHE_LOG");return v?std::stol(v):0L;}();
        static double win_n=0,win_hit=0,win_wall=0,win_calls=0;static long windows=0;
        win_n+=n;win_hit+=hit;win_wall+=wall;win_calls++;
        if(every>0&&b->index==0&&++windows%every==0) {
            auto &pm=pool_manager();
            std::fprintf(stderr,"[xtier] XTier cache: %ld windows, hit %.3f, %.1f distinct experts and %.0f us CPU per layer call; %ld swaps in %ld rounds so far, %ld prompts\n",
                         every,win_hit/std::max(win_n,1.0),win_n/std::max(win_calls,1.0),win_wall/std::max(win_calls,1.0),pm.swaps,pm.rounds,pm.prompts);
            win_n=win_hit=win_wall=win_calls=0;
        }
    }
    if(std::FILE *f=trace_file()) {
        static uint32_t seq=0;
        TraceHead h{uint64_t(real.tv_sec)*1000000000ull+uint64_t(real.tv_nsec),uint32_t(wall*1e3),
                    uint32_t(phases[0]*1e3),uint32_t(phases[1]*1e3),uint16_t(b->index),uint8_t(m),uint8_t(k),
                    int16_t(rc),uint8_t(active),0,seq++};
        std::fwrite(&h,sizeof h,1,f);std::fwrite(pool?s->orig:s->ids,4,size_t(m)*k,f);
    }
}

Dispatcher::Dispatcher() {
    int cpu=80;
    if(const char *v=std::getenv("XTIER_PROBE_DISPATCH_CPU"))cpu=std::stoi(v);
    // The caller's own mask may exclude this CPU (e.g. a pinned scheduler thread);
    // the dispatcher thread sets its affinity itself and reports rejection below.
    if(cpu<0||cpu>=CPU_SETSIZE)throw std::runtime_error("Dispatcher CPU unavailable");
    long idle_ms=100,idle_sleep_us=50;
    if(const char *v=std::getenv("XTIER_DISPATCH_IDLE_MS"))idle_ms=std::stol(v);
    if(const char *v=std::getenv("XTIER_DISPATCH_IDLE_SLEEP_US"))idle_sleep_us=std::stol(v);
    std::atomic<int> ready{0};
    worker=std::thread([this,cpu,idle_ms,idle_sleep_us,&ready]{
        cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
        if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set)){ready.store(-1);return;}
        ready.store(1,std::memory_order_release);
        auto active=std::chrono::steady_clock::now();
        bool trace_dirty=false;
        const timespec nap{0,idle_sleep_us*1000};
        while(!quit.load(std::memory_order_relaxed)) {
            bool served=false;
            {
                // Serializes registration/removal with every dereference: no
                // stale slots during close, and no locks held by a CUDA callback.
                std::lock_guard<std::mutex> lock(mutex);
                for(auto *s:slots)if(__atomic_load_n(s->mailbox,__ATOMIC_ACQUIRE)==1) {
                    busy_since.store(std::chrono::steady_clock::now().time_since_epoch().count(),std::memory_order_relaxed);
                    execute(s);
                    busy_since.store(0,std::memory_order_relaxed);
                    // x86 coherent WB host memory. Publish only after all worker
                    // output stores and stats complete; GPU then copies output.
                    __atomic_store_n(s->mailbox,2,__ATOMIC_RELEASE);
                    served=true;
                }
            }
            if(idle_ms>0) {
                // Full-rate polling while tokens are flowing; after idle_ms without a
                // request, poll every idle_sleep_us instead of burning the SMT sibling.
                auto now=std::chrono::steady_clock::now();
                if(served){active=now;trace_dirty=true;}
                else if(now-active>std::chrono::milliseconds(idle_ms)) {
                    if(trace_dirty){if(std::FILE *f=trace_file())std::fflush(f);trace_dirty=false;}
                    nanosleep(&nap,nullptr);continue;
                }
            }
            _mm_pause();
        }
    });
    while(!ready.load(std::memory_order_acquire))_mm_pause();
    if(ready.load()<0){worker.join();throw std::runtime_error("Dispatcher affinity rejected");}
    long timeout_s=60;
    if(const char *v=std::getenv("XTIER_CPU_TIMEOUT_S"))timeout_s=std::stol(v);
    if(timeout_s>0)watchdog=std::thread([this,timeout_s]{
        // The GPU stream waits on the mailbox without a timeout. Turn a stuck CPU call
        // or an unserved request into an immediate, logged abort instead of a silent
        // hang until the framework watchdog (default 1200 s).
        const int64_t limit=int64_t(timeout_s)*1000000000;
        while(!quit.load(std::memory_order_relaxed)) {
            for(int i=0;i<10&&!quit.load(std::memory_order_relaxed);i++)std::this_thread::sleep_for(std::chrono::milliseconds(100));
            int64_t now=std::chrono::steady_clock::now().time_since_epoch().count();
            int64_t busy=busy_since.load(std::memory_order_relaxed);
            const char *what=nullptr;
            if(busy&&now-busy>limit)what="a CPU expert call is still running";
            else if(std::unique_lock<std::mutex> lock(mutex,std::try_to_lock);lock.owns_lock()) {
                for(auto *s:slots) {
                    if(__atomic_load_n(s->mailbox,__ATOMIC_ACQUIRE)!=1){s->pending_since=0;continue;}
                    if(!s->pending_since)s->pending_since=now;
                    else if(now-s->pending_since>limit)what="a GPU request is waiting unserved";
                }
            }
            if(what) {
                std::fprintf(stderr,"[xtier] FATAL: %s for more than %ld s (XTIER_CPU_TIMEOUT_S); aborting\n",what,timeout_s);
                std::fflush(stderr);std::abort();
            }
        }
    });
}
Dispatcher::~Dispatcher(){quit.store(true);if(worker.joinable())worker.join();if(watchdog.joinable())watchdog.join();}
void Dispatcher::add(Slot *s){std::lock_guard<std::mutex> lock(mutex);slots.push_back(s);}
void Dispatcher::remove(Slot *s){std::lock_guard<std::mutex> lock(mutex);slots.erase(std::remove(slots.begin(),slots.end(),s),slots.end());}
std::shared_ptr<Dispatcher> get_dispatcher() {
    static std::mutex mutex;static std::weak_ptr<Dispatcher> existing;
    std::lock_guard<std::mutex> lock(mutex);auto p=existing.lock();
    if(!p){p=std::make_shared<Dispatcher>();existing=p;}return p;
}

// Mailbox handshake as stream memory operations (one batch node when captured).
// parts: 1 = request (write 1), 2 = wait for the CPU (wait 2, write 0), 3 = both.
void notify_and_wait(cudaStream_t stream,Slot *s,int parts=3) {
    CUstreamBatchMemOpParams all[3]{};
    all[0].writeValue.operation=CU_STREAM_MEM_OP_WRITE_VALUE_32;
    all[0].writeValue.address=s->device_mailbox;all[0].writeValue.value=1;
    all[1].waitValue.operation=CU_STREAM_MEM_OP_WAIT_VALUE_32;
    all[1].waitValue.address=s->device_mailbox;all[1].waitValue.value=2;
    all[1].waitValue.flags=CU_STREAM_WAIT_VALUE_EQ;
    all[2].writeValue.operation=CU_STREAM_MEM_OP_WRITE_VALUE_32;
    all[2].writeValue.address=s->device_mailbox;all[2].writeValue.value=0;
    CUstreamBatchMemOpParams *ops=parts==2?all+1:all;
    const unsigned nops=parts==1?1:parts==2?2:3;
    cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;
    const cudaGraphNode_t *deps=nullptr;size_t count=0;
    check(cudaStreamGetCaptureInfo(stream,&status,nullptr,&graph,&deps,nullptr,&count));
    if(status==cudaStreamCaptureStatusNone) {
        driver(cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream),nops,ops,0));return;
    }
    if(status!=cudaStreamCaptureStatusActive)throw std::runtime_error("Invalid capture state");
    CUDA_BATCH_MEM_OP_NODE_PARAMS params{};driver(cuCtxGetCurrent(&params.ctx));
    params.count=nops;params.paramArray=ops;
    CUgraphNode node;
    driver(cuGraphAddBatchMemOpNode(&node,reinterpret_cast<CUgraph>(graph),
                                  reinterpret_cast<const CUgraphNode*>(deps),count,&params));
    cudaGraphNode_t runtime_node=reinterpret_cast<cudaGraphNode_t>(node);
    check(cudaStreamUpdateCaptureDependencies(stream,&runtime_node,nullptr,1,cudaStreamSetCaptureDependencies));
}
struct PromptCounts {const float *host;int layers,stride;};
void CUDART_CB prompt_counts(void *p) noexcept {   // (a host function: no CUDA calls)
    auto *c=static_cast<PromptCounts*>(p);
    try{pool_manager().prompt(c->host,c->layers,c->stride);}catch(...){}
    delete c;
}
}
extern "C" {
const char *xtier_mailbox_error(){return error.c_str();}
void *xtier_mailbox_create(void *plan) {
    try {return new Bridge(plan);}catch(const std::exception&e){error=e.what();return nullptr;}
}
int xtier_mailbox_prepare(void *bridge,void *stream,int m,int k,int shadow) {
    try {
        if(!bridge||m<1||m>MAX_SLOT_M||k<1||k>10)throw std::runtime_error("Invalid bridge shape");
        auto*b=static_cast<Bridge*>(bridge);std::lock_guard<std::mutex> lock(b->slot_mutex);
        b->find(static_cast<cudaStream_t>(stream),m,k,shadow!=0);return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_enqueue(void *bridge,void *stream,int m,int k,const void *x,const void *ids,
                       const void *routing,void *output,int shadow) {
    try {
        if(!bridge||!x||!ids||!routing||!output||m<1||m>MAX_SLOT_M||k<1||k>10)
            throw std::runtime_error("Invalid enqueue arguments");
        auto*b=static_cast<Bridge*>(bridge);
        if(b->failed.load(std::memory_order_acquire))throw std::runtime_error("Prior CPU callback failed");
        auto st=static_cast<cudaStream_t>(stream);std::lock_guard<std::mutex> lock(b->slot_mutex);
        auto*s=b->find(st,m,k,shadow!=0);
        check(cudaMemcpyAsync(s->x,x,m*s->H*2,cudaMemcpyDeviceToHost,st));
        check(cudaMemcpyAsync(s->ids,ids,m*k*4,cudaMemcpyDeviceToHost,st));
        check(cudaMemcpyAsync(s->routing,routing,m*k*4,cudaMemcpyDeviceToHost,st));
        if(shadow)check(cudaMemcpyAsync(s->reference,output,m*s->H*4,cudaMemcpyDeviceToHost,st));
        notify_and_wait(st,s);
        if(!shadow)check(cudaMemcpyAsync(output,s->out,m*s->H*4,cudaMemcpyHostToDevice,st));
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_slot(void *bridge,void *stream,int m,int k,void **pointers) {
    try {
        if(!bridge||!pointers||m<1||m>MAX_SLOT_M||k<1||k>10)throw std::runtime_error("Invalid slot query");
        auto*b=static_cast<Bridge*>(bridge);std::lock_guard<std::mutex> lock(b->slot_mutex);
        auto*s=b->find(static_cast<cudaStream_t>(stream),m,k,false);
        auto *base=static_cast<char*>(s->storage);
        void *host[4]={s->x,s->ids,s->routing,s->out};
        for(int j=0;j<4;j++)pointers[j]=s->device_storage+(static_cast<char*>(host[j])-base);
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_notify(void *bridge,void *stream,int m,int k) {
    try {
        if(!bridge||m<1||m>MAX_SLOT_M||k<1||k>10)throw std::runtime_error("Invalid notify arguments");
        auto*b=static_cast<Bridge*>(bridge);
        if(b->failed.load(std::memory_order_acquire))throw std::runtime_error("Prior CPU callback failed");
        auto st=static_cast<cudaStream_t>(stream);std::lock_guard<std::mutex> lock(b->slot_mutex);
        notify_and_wait(st,b->find(st,m,k,false));
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_slot2(void *bridge,void *stream,int m,int k,void **pointers) {
    try {
        if(!bridge||!pointers||m<1||m>MAX_SLOT_M||k<1||k>10)throw std::runtime_error("Invalid slot query");
        auto*b=static_cast<Bridge*>(bridge);std::lock_guard<std::mutex> lock(b->slot_mutex);
        auto*s=b->find(static_cast<cudaStream_t>(stream),m,k,false);
        auto *base=static_cast<char*>(s->storage);
        void *host[5]={s->x,s->ids,s->routing,s->out,s->orig};
        for(int j=0;j<5;j++)pointers[j]=s->device_storage+(static_cast<char*>(host[j])-base);
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_signal(void *bridge,void *stream,int m,int k,int part) {
    try {
        if(!bridge||m<1||m>MAX_SLOT_M||k<1||k>10||part<1||part>2)throw std::runtime_error("Invalid signal arguments");
        auto*b=static_cast<Bridge*>(bridge);
        if(b->failed.load(std::memory_order_acquire))throw std::runtime_error("Prior CPU callback failed");
        auto st=static_cast<cudaStream_t>(stream);std::lock_guard<std::mutex> lock(b->slot_mutex);
        notify_and_wait(st,b->find(st,m,k,false),part);
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_pool_enable(void *bridge,int on) {
    if(!bridge)return -1;
    static_cast<Bridge*>(bridge)->pool.store(on!=0,std::memory_order_release);return 0;
}
int xtier_mailbox_pool_ring(void **pointers) {
    try {
        auto &m=pool_manager();std::lock_guard<std::mutex> lock(m.mu);m.ensure_ring();
        pointers[0]=reinterpret_cast<void*>(m.dev_ring);pointers[1]=reinterpret_cast<void*>(m.dev_seqs);
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_pool_attach(void *bridge,int layer,int slots,int experts,void **arena,const int32_t *resident) {
    try {
        if(!bridge||layer<0||layer>4095||slots<1||experts<1||!arena||!resident)throw std::runtime_error("Invalid pool attach");
        auto*b=static_cast<Bridge*>(bridge);
        auto L=std::make_unique<PoolLayer>();
        int shape[4];if(xtier_shape(b->plan,shape)<0)throw std::runtime_error(xtier_last_error());
        if(shape[2]!=experts)throw std::runtime_error("Pool attach: expert count mismatch");
        L->bridge=b;L->layer=layer;L->S=slots;L->E=experts;L->H=shape[0];L->I=shape[1];
        for(int j=0;j<4;j++)L->arena[j]=static_cast<char*>(arena[j]);
        L->nodes=xtier_nodes(b->plan);
        if(L->nodes<1||L->nodes>16)throw std::runtime_error("Pool attach: invalid worker node count");
        for(int node=0;node<L->nodes;node++) {
            void *ptr[6];size_t bytes[6];
            if(xtier_shard(b->plan,node,ptr,bytes)<0)throw std::runtime_error(xtier_last_error());
            const size_t scale_width=bytes[2]/(size_t(L->E)*(L->I/L->nodes));
            if(!scale_width || size_t(L->H)%scale_width)throw std::runtime_error("Invalid scale shard width");
            const int group=L->H/scale_width;
            if((group!=16&&group!=32)||(node&&group!=L->group))throw std::runtime_error("Invalid scale group");
            L->group=group;
            for(int j=0;j<6;j++)L->shard[node][j]=static_cast<const char*>(ptr[j]);
        }
        L->resident.assign(resident,resident+slots);L->slot_of.assign(experts,-1);L->busy.assign(slots,0);
        L->usage.assign(experts,0.f);
        for(int s=0;s<slots;s++)if(L->resident[s]>=0&&L->resident[s]<experts)L->slot_of[L->resident[s]]=s;
        auto &m=pool_manager();std::lock_guard<std::mutex> lock(m.mu);m.ensure_ring();
        if(m.layers.size()<=size_t(layer))m.layers.resize(layer+1);
        m.layers[layer]=std::move(L);b->pool_layer=layer;
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_pool_start(int every,int budget,float decay,float min_gain) {
    try {
        auto &m=pool_manager();std::lock_guard<std::mutex> lock(m.mu);
        if(every<1||budget<0||!(decay>0.f&&decay<=1.f))throw std::runtime_error("Invalid pool policy");
        m.every=every;m.budget=budget;m.decay=decay;m.min_gain=min_gain;m.start();
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
// the prompt counts in pinned host memory once the stream reaches this point (after their copy): blended in then
int xtier_mailbox_pool_prompt(void *stream,const float *host,int layers,int stride) {
    try {
        if(!host||layers<1||stride<1)throw std::runtime_error("Invalid pool prompt counts");
        check(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream),prompt_counts,new PromptCounts{host,layers,stride}));
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_pool_warm(float keep,float windows,int rounds,int budget) {
    try {
        if(!(keep>=0.f&&keep<=1.f)||!(windows>0.f)||rounds<0||budget<0)throw std::runtime_error("Invalid pool warm-up");
        auto &m=pool_manager();std::lock_guard<std::mutex> lock(m.mu);
        m.warm_keep=keep;m.warm_windows=windows;m.warm_rounds=rounds;m.warm_budget=budget;
        return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
int xtier_mailbox_pool_stats(void *bridge,double *out) {
    if(!bridge||!out)return -1;
    auto*b=static_cast<Bridge*>(bridge);
    std::lock_guard<std::mutex> lock(b->stats_mutex);std::copy(b->pool_stats.begin(),b->pool_stats.end(),out);
    return 0;
}
int xtier_mailbox_stats(void *bridge,double *out) {
    if(!bridge||!out)return -1;
    auto*b=static_cast<Bridge*>(bridge);
    std::lock_guard<std::mutex> lock(b->stats_mutex);std::copy(b->stats.begin(),b->stats.end(),out);
    return b->failed.load(std::memory_order_acquire) ? -1 : 0;
}
int xtier_mailbox_timing(void *bridge,double *out) {
    if(!bridge||!out)return -1;
    auto*b=static_cast<Bridge*>(bridge);
    std::lock_guard<std::mutex> lock(b->stats_mutex);std::copy(b->timing.begin(),b->timing.end(),out);
    return 0;
}
int xtier_mailbox_destroy(void *bridge) {
    try {
        if(!bridge)return 0;
        // Caller must destroy graphs referring to this bridge before close.
        // Synchronize before releasing pinned memory and the retained plan.
        check(cudaDeviceSynchronize());delete static_cast<Bridge*>(bridge);return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
}
