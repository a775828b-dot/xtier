// Multi-drive O_DIRECT + io_uring KV cache tier for XTier (libxtier_kvstore.so).
// Design: volatile in-memory sharded index + per-drive preallocated pool.dat ring divided into 256 MiB regions,
// CLOCK hot-entry relocation with >=2 reserved empty regions, R1 synchronous write completion, and R2 lock-ordered
// per-region in-flight read reference counting with non-blocking DriveWorker draining.
#include "xtier_kvstore.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <liburing.h>
#include <limits.h>
#include <memory>
#include <mutex>
#include <sched.h>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <sys/eventfd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define XKV_API __attribute__((visibility("default")))

namespace {

constexpr size_t kAlign = 4096;
constexpr int kNumShards = 16;
constexpr uint64_t kDefaultRegionBytes = 256ULL * 1024 * 1024;
constexpr uint32_t kDefaultWriteQD = 4;
constexpr uint32_t kDefaultReadQD = 64;
constexpr uint32_t kDefaultBounceBytes = 32U * 1024 * 1024;
constexpr uint32_t kDefaultRelocatePct = 50;
constexpr uint32_t kDefaultRingDepth = 128;
constexpr size_t kRelocChunkBytes = 16ULL * 1024 * 1024;
constexpr uint64_t kWakeUserData = ~0ULL;

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }
void set_errno(const char* what, const std::string& detail, int err) {
    g_last_error = std::string(what) + (detail.empty() ? "" : " (" + detail + ")") + ": " + std::strerror(err);
}

inline size_t round_up_4k(size_t n) { return (n + kAlign - 1) & ~(kAlign - 1); }

struct IndexKey {
    uint64_t lo;
    uint64_t hi;
    uint16_t pool_id;

    bool operator==(const IndexKey& o) const noexcept {
        return lo == o.lo && hi == o.hi && pool_id == o.pool_id;
    }
};

struct IndexKeyHash {
    size_t operator()(const IndexKey& k) const noexcept {
        uint64_t h = k.lo ^ (k.hi * 0x9e3779b97f4a7c15ULL);
        h ^= (uint64_t(k.pool_id) << 48) ^ (h >> 30);
        h *= 0xbf58476d1ce4e5b9ULL;
        return size_t(h ^ (h >> 27));
    }
};

inline int shard_of(const IndexKey& k) noexcept { return int((k.lo >> 4) & (kNumShards - 1)); }
inline int drive_of(const xtier_kv_key_t& k, int num_drives) noexcept {
    return int(k.lo % uint64_t(num_drives));
}

struct IndexValue {
    uint8_t drive;
    uint16_t region;
    uint32_t bytes;
    uint64_t offset;
    uint64_t gen;
    std::atomic<uint8_t> hot{0};

    IndexValue(uint8_t d, uint16_t r, uint32_t b, uint64_t off, uint64_t g)
        : drive(d), region(r), bytes(b), offset(off), gen(g), hot(0) {}
};

struct IndexShard {
    std::shared_mutex mu;
    std::unordered_map<IndexKey, IndexValue, IndexKeyHash> map;
};

struct RegionEntry {
    IndexKey key;
    uint64_t offset;
    uint32_t bytes;
    uint64_t gen;
};

struct Region {
    std::vector<RegionEntry> entries;  // touched only while holding Drive::write_mu
    std::atomic<uint32_t> in_flight_reads{0};
    uint64_t gen{1};
};

// Simple 4 KiB-aligned first-fit arena for unaligned O_DIRECT bounce I/Os inside one DriveWorker.
class BounceArena {
public:
    BounceArena() = default;
    ~BounceArena() {
        if (base_) std::free(base_);
    }
    BounceArena(const BounceArena&) = delete;
    BounceArena& operator=(const BounceArena&) = delete;

    bool init(size_t bytes) {
        size_ = round_up_4k(std::max<size_t>(bytes, kAlign * 16));
        void* p = nullptr;
        if (posix_memalign(&p, kAlign, size_) != 0 || !p) return false;
        base_ = static_cast<uint8_t*>(p);
        free_.push_back({0, size_});
        return true;
    }

    uint8_t* alloc(size_t n, bool& heap_alloc) {
        heap_alloc = false;
        size_t need = round_up_4k(n);
        if (need > size_) {
            void* p = nullptr;
            if (posix_memalign(&p, kAlign, need) != 0 || !p) return nullptr;
            heap_alloc = true;
            return static_cast<uint8_t*>(p);
        }
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i].len >= need) {
                size_t off = free_[i].off;
                if (free_[i].len == need) {
                    free_.erase(free_.begin() + ptrdiff_t(i));
                } else {
                    free_[i].off += need;
                    free_[i].len -= need;
                }
                return base_ + off;
            }
        }
        return nullptr;
    }

    void release(uint8_t* ptr, size_t n, bool heap_alloc) {
        if (!ptr) return;
        if (heap_alloc) {
            std::free(ptr);
            return;
        }
        size_t off = size_t(ptr - base_);
        size_t len = round_up_4k(n);
        auto it = std::lower_bound(free_.begin(), free_.end(), off,
                                   [](const Span& s, size_t o) { return s.off < o; });
        it = free_.insert(it, {off, len});
        if (it + 1 != free_.end() && it->off + it->len == (it + 1)->off) {
            it->len += (it + 1)->len;
            free_.erase(it + 1);
        }
        if (it != free_.begin() && (it - 1)->off + (it - 1)->len == it->off) {
            (it - 1)->len += it->len;
            free_.erase(it);
        }
    }

private:
    struct Span {
        size_t off;
        size_t len;
    };
    uint8_t* base_{nullptr};
    size_t size_{0};
    std::vector<Span> free_;
};

struct IoBatch;

struct IoOp {
    bool is_write{false};
    bool track_region_read{false};
    uint16_t region{0};
    int item_idx{-1};
    uint64_t offset{0};
    size_t logical_bytes{0};
    size_t aligned_bytes{0};
    std::vector<struct iovec> iovs;

    bool use_bounce{false};
    bool bounce_heap{false};
    uint8_t* bounce_ptr{nullptr};
    size_t cur_done{0};
    size_t cur_iov_idx{0};
    size_t cur_iov_off{0};
    bool ok{false};
    IoBatch* batch{nullptr};
};

struct IoBatch {
    std::mutex mu;
    std::condition_variable cv;
    size_t remaining{0};
    std::vector<IoOp> ops;

    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return remaining == 0; });
    }
};

void copy_iovs_to_buf(const std::vector<struct iovec>& iovs, uint8_t* dst, size_t total, size_t aligned) {
    size_t pos = 0;
    for (const auto& v : iovs) {
        if (v.iov_len) {
            std::memcpy(dst + pos, v.iov_base, v.iov_len);
            pos += v.iov_len;
        }
    }
    if (aligned > total) {
        std::memset(dst + total, 0, aligned - total);
    }
}

void copy_buf_to_iovs(const uint8_t* src, const std::vector<struct iovec>& iovs) {
    size_t pos = 0;
    for (const auto& v : iovs) {
        if (v.iov_len) {
            std::memcpy(v.iov_base, src + pos, v.iov_len);
            pos += v.iov_len;
        }
    }
}

bool all_iovs_direct_aligned(const std::vector<struct iovec>& iovs, size_t logical, size_t aligned) {
    if (iovs.empty() || logical != aligned || iovs.size() > size_t(IOV_MAX)) return false;
    for (const auto& v : iovs) {
        if ((uintptr_t(v.iov_base) & (kAlign - 1)) != 0) return false;
        if ((v.iov_len & (kAlign - 1)) != 0) return false;
    }
    return true;
}

int detect_fd_numa_node(int fd) {
    struct stat st {};
    if (fstat(fd, &st) != 0) return -1;
    unsigned maj = major(st.st_dev);
    unsigned min = minor(st.st_dev);
    std::string base = "/sys/dev/block/" + std::to_string(maj) + ":" + std::to_string(min);
    const std::string candidates[] = {
        base + "/device/numa_node",
        base + "/../device/numa_node",
    };
    for (const auto& path : candidates) {
        std::ifstream f(path);
        int node = -1;
        if (f >> node && node >= 0) return node;
    }
    return -1;
}

std::vector<int> parse_cpulist(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto dash = tok.find('-');
        if (dash == std::string::npos) {
            if (!tok.empty()) out.push_back(std::stoi(tok));
        } else {
            int a = std::stoi(tok.substr(0, dash));
            int b = std::stoi(tok.substr(dash + 1));
            for (int c = a; c <= b; ++c) out.push_back(c);
        }
    }
    return out;
}

void pin_worker_to_numa(int numa_node) {
    if (numa_node < 0) return;
    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (sched_getaffinity(0, sizeof(cur), &cur) != 0) return;

    std::ifstream f("/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist");
    std::string line;
    if (!std::getline(f, line) || line.empty()) return;

    cpu_set_t target;
    CPU_ZERO(&target);
    int matched = 0;
    for (int cpu : parse_cpulist(line)) {
        if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &cur)) {
            CPU_SET(cpu, &target);
            matched++;
        }
    }
    if (matched > 0) {
        sched_setaffinity(0, sizeof(target), &target);
    }
}

struct Drive {
    int id{0};
    int fd{-1};
    int efd{-1};
    int numa_node{-1};
    std::string path;

    uint64_t capacity_bytes{0};
    uint64_t region_bytes{0};
    uint32_t num_regions{0};
    uint32_t write_qd{kDefaultWriteQD};
    uint32_t read_qd{kDefaultReadQD};
    uint32_t ring_depth{kDefaultRingDepth};
    uint32_t relocate_pct{kDefaultRelocatePct};
    bool pin_numa{true};

    // Write/recycle coordinator state (protected exclusively by write_mu; DriveWorker NEVER locks write_mu!)
    std::mutex write_mu;
    uint32_t write_region{0};
    uint64_t write_pos{0};
    uint32_t evict_region{0};
    uint32_t used_regions{1};
    std::unique_ptr<Region[]> regions;
    uint8_t* reloc_buf{nullptr};
    size_t reloc_buf_bytes{0};

    // Region in-flight read drain notification
    std::mutex evict_mu;
    std::condition_variable evict_cv;

    // DriveWorker queues
    std::mutex q_mu;
    std::deque<IoOp*> read_q;
    std::deque<IoOp*> write_q;
    bool stop{false};

    // DriveWorker thread & ring
    struct io_uring ring {};
    bool ring_inited{false};
    BounceArena bounce;
    uint64_t efd_buf{0};
    bool wake_armed{false};
    std::thread worker;

    // Per-drive atomic counters aggregated into stats
    std::atomic<uint64_t> direct_ios{0};
    std::atomic<uint64_t> bounce_ios{0};
    std::atomic<uint64_t> regions_recycled{0};
    std::atomic<uint64_t> hot_relocated{0};
    std::atomic<uint64_t> hot_dropped_cap{0};
    std::atomic<uint64_t> evict_wait_count{0};

    ~Drive() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(q_mu);
                stop = true;
            }
            wake_worker();
            worker.join();
        }
        if (ring_inited) io_uring_queue_exit(&ring);
        if (efd >= 0) ::close(efd);
        if (fd >= 0) {
            ::flock(fd, LOCK_UN);
            ::close(fd);
        }
        if (reloc_buf) std::free(reloc_buf);
    }

    void wake_worker() {
        if (efd >= 0) {
            uint64_t one = 1;
            ssize_t r = ::write(efd, &one, sizeof(one));
            (void)r;
        }
    }

    void enqueue_batch(IoBatch* batch) {
        if (batch->ops.empty()) return;
        batch->remaining = batch->ops.size();
        {
            std::lock_guard<std::mutex> lk(q_mu);
            for (auto& op : batch->ops) {
                op.batch = batch;
                if (op.is_write) {
                    write_q.push_back(&op);
                } else {
                    read_q.push_back(&op);
                }
            }
        }
        wake_worker();
    }

    void release_region_read(uint16_t reg) {
        uint32_t prev = regions[reg].in_flight_reads.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            std::lock_guard<std::mutex> lk(evict_mu);
            evict_cv.notify_all();
        }
    }

    void finish_op(IoOp* op, bool ok) {
        op->ok = ok;
        if (op->use_bounce) {
            if (!op->is_write && ok) {
                copy_buf_to_iovs(op->bounce_ptr, op->iovs);
            }
            bounce.release(op->bounce_ptr, op->aligned_bytes, op->bounce_heap);
            op->bounce_ptr = nullptr;
        }
        if (op->track_region_read) {
            release_region_read(op->region);
        }
        IoBatch* b = op->batch;
        std::lock_guard<std::mutex> lk(b->mu);
        if (--b->remaining == 0) {
            b->cv.notify_all();
        }
    }

    void arm_wake_sqe() {
        if (wake_armed) return;
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) return;
        io_uring_prep_read(sqe, efd, &efd_buf, sizeof(efd_buf), 0);
        io_uring_sqe_set_data64(sqe, kWakeUserData);
        wake_armed = true;
    }

    bool prep_op_sqe(IoOp* op) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) return false;
        uint64_t off = op->offset + op->cur_done;
        size_t rem = op->aligned_bytes - op->cur_done;
        if (op->use_bounce) {
            if (op->is_write) {
                io_uring_prep_write(sqe, fd, op->bounce_ptr + op->cur_done, unsigned(rem), off);
            } else {
                io_uring_prep_read(sqe, fd, op->bounce_ptr + op->cur_done, unsigned(rem), off);
            }
        } else {
            struct iovec* vec = op->iovs.data() + op->cur_iov_idx;
            unsigned cnt = unsigned(op->iovs.size() - op->cur_iov_idx);
            if (cnt == 1) {
                uint8_t* ptr = static_cast<uint8_t*>(vec[0].iov_base) + op->cur_iov_off;
                size_t len = vec[0].iov_len - op->cur_iov_off;
                if (op->is_write) {
                    io_uring_prep_write(sqe, fd, ptr, unsigned(len), off);
                } else {
                    io_uring_prep_read(sqe, fd, ptr, unsigned(len), off);
                }
            } else {
                if (op->cur_iov_off > 0) {
                    vec[0].iov_base = static_cast<uint8_t*>(vec[0].iov_base) + op->cur_iov_off;
                    vec[0].iov_len -= op->cur_iov_off;
                    op->cur_iov_off = 0;
                }
                if (op->is_write) {
                    io_uring_prep_writev(sqe, fd, vec, cnt, off);
                } else {
                    io_uring_prep_readv(sqe, fd, vec, cnt, off);
                }
            }
        }
        io_uring_sqe_set_data(sqe, op);
        return true;
    }

    void advance_direct_iovs(IoOp* op, size_t got) {
        while (got > 0 && op->cur_iov_idx < op->iovs.size()) {
            size_t avail = op->iovs[op->cur_iov_idx].iov_len - op->cur_iov_off;
            if (got >= avail) {
                got -= avail;
                op->cur_iov_idx++;
                op->cur_iov_off = 0;
            } else {
                op->cur_iov_off += got;
                got = 0;
            }
        }
    }

    void worker_loop() {
        if (pin_numa) pin_worker_to_numa(numa_node);

        std::deque<IoOp*> local_reads;
        std::deque<IoOp*> local_writes;
        uint32_t inflight_reads = 0;
        uint32_t inflight_writes = 0;
        uint32_t max_inflight_total = (ring_depth > 8) ? (ring_depth - 4) : 4;
        uint32_t eff_read_qd = (read_qd > 0) ? std::min(read_qd, max_inflight_total) : max_inflight_total;
        uint32_t eff_write_qd = std::max<uint32_t>(1, std::min(write_qd, max_inflight_total));

        for (;;) {
            bool stopping = false;
            {
                std::lock_guard<std::mutex> lk(q_mu);
                while (!read_q.empty()) {
                    local_reads.push_back(read_q.front());
                    read_q.pop_front();
                }
                while (!write_q.empty()) {
                    local_writes.push_back(write_q.front());
                    write_q.pop_front();
                }
                stopping = stop;
            }

            if (stopping && local_reads.empty() && local_writes.empty() &&
                inflight_reads == 0 && inflight_writes == 0) {
                break;
            }

            auto try_submit_one = [&](std::deque<IoOp*>& q, uint32_t& inflight_cnt, uint32_t qd_cap) -> bool {
                if (q.empty() || inflight_cnt >= qd_cap || (inflight_reads + inflight_writes) >= max_inflight_total) {
                    return false;
                }
                IoOp* op = q.front();
                bool direct = all_iovs_direct_aligned(op->iovs, op->logical_bytes, op->aligned_bytes);
                if (!direct) {
                    bool heap_alloc = false;
                    uint8_t* ptr = bounce.alloc(op->aligned_bytes, heap_alloc);
                    if (!ptr) return false;
                    op->use_bounce = true;
                    op->bounce_heap = heap_alloc;
                    op->bounce_ptr = ptr;
                    if (op->is_write) {
                        copy_iovs_to_buf(op->iovs, ptr, op->logical_bytes, op->aligned_bytes);
                    }
                    bounce_ios.fetch_add(1, std::memory_order_relaxed);
                } else {
                    op->use_bounce = false;
                    direct_ios.fetch_add(1, std::memory_order_relaxed);
                }
                if (!prep_op_sqe(op)) {
                    if (op->use_bounce) {
                        bounce.release(op->bounce_ptr, op->aligned_bytes, op->bounce_heap);
                        op->bounce_ptr = nullptr;
                        op->use_bounce = false;
                    }
                    return false;
                }
                q.pop_front();
                inflight_cnt++;
                return true;
            };

            while (try_submit_one(local_reads, inflight_reads, eff_read_qd)) {}
            if (local_reads.empty()) {
                while (try_submit_one(local_writes, inflight_writes, eff_write_qd)) {}
            }

            arm_wake_sqe();

            int ret = io_uring_submit_and_wait(&ring, 1);
            if (ret < 0 && ret != -EINTR) {
                continue;
            }

            struct io_uring_cqe* cqe = nullptr;
            unsigned head = 0;
            unsigned count = 0;
            std::vector<IoOp*> retry_ops;
            io_uring_for_each_cqe(&ring, head, cqe) {
                count++;
                uint64_t ud = io_uring_cqe_get_data64(cqe);
                int res = cqe->res;
                if (ud == kWakeUserData) {
                    wake_armed = false;
                    continue;
                }
                IoOp* op = reinterpret_cast<IoOp*>(ud);
                if (res == -EINTR || res == -EAGAIN) {
                    retry_ops.push_back(op);
                    continue;
                }
                if (res > 0) {
                    size_t got = size_t(res);
                    op->cur_done += got;
                    if (op->cur_done < op->aligned_bytes) {
                        if (!op->use_bounce) advance_direct_iovs(op, got);
                        retry_ops.push_back(op);
                        continue;
                    }
                    if (op->is_write) inflight_writes--;
                    else inflight_reads--;
                    finish_op(op, true);
                } else {
                    if (op->is_write) inflight_writes--;
                    else inflight_reads--;
                    finish_op(op, false);
                }
            }
            if (count > 0) io_uring_cq_advance(&ring, count);

            for (IoOp* op : retry_ops) {
                if (!prep_op_sqe(op)) {
                    io_uring_submit(&ring);
                    if (!prep_op_sqe(op)) {
                        if (op->is_write) inflight_writes--;
                        else inflight_reads--;
                        finish_op(op, false);
                    }
                }
            }
        }
    }
};

}  // namespace

struct xtier_kv_store {
    std::vector<std::unique_ptr<Drive>> drives;
    IndexShard shards[kNumShards];

    std::atomic<uint64_t> live_entries{0};
    std::atomic<uint64_t> live_bytes{0};
    std::atomic<uint64_t> puts_ok{0};
    std::atomic<uint64_t> puts_skipped_dup{0};
    std::atomic<uint64_t> puts_failed{0};
    std::atomic<uint64_t> gets_hit{0};
    std::atomic<uint64_t> gets_miss{0};
    std::atomic<uint64_t> gets_failed{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint32_t> test_read_delay_us{0};

    // Caller MUST hold drive->write_mu. drive->write_region MUST be newly advanced & empty (write_pos == 0).
    void recycle_oldest_region(Drive* d) {
        uint32_t ev = d->evict_region;
        Region& ev_reg = d->regions[ev];
        uint32_t wr = d->write_region;
        Region& wr_reg = d->regions[wr];
        const uint64_t max_reloc = (d->region_bytes * d->relocate_pct) / 100ULL;

        struct RelocItem {
            IndexKey key;
            uint64_t old_offset;
            uint32_t bytes;
            uint64_t old_gen;
            size_t aligned;
        };
        std::vector<RelocItem> reloc;
        uint64_t reloc_bytes = 0;

        // Step A: drop cold (or over-cap hot) entries; keep hot entries live in index during relocation.
        for (const auto& e : ev_reg.entries) {
            IndexShard& sh = shards[shard_of(e.key)];
            std::unique_lock<std::shared_mutex> lk(sh.mu);
            auto it = sh.map.find(e.key);
            if (it == sh.map.end()) continue;
            const IndexValue& v = it->second;
            if (v.drive != uint8_t(d->id) || v.region != uint16_t(ev) || v.offset != e.offset || v.gen != e.gen) {
                continue;
            }
            size_t al = round_up_4k(e.bytes);
            bool is_hot = v.hot.load(std::memory_order_relaxed) != 0;
            if (is_hot && reloc_bytes + al <= max_reloc && d->write_pos + reloc_bytes + al <= d->region_bytes) {
                reloc.push_back({e.key, e.offset, e.bytes, e.gen, al});
                reloc_bytes += al;
            } else {
                if (is_hot) d->hot_dropped_cap.fetch_add(1, std::memory_order_relaxed);
                live_entries.fetch_sub(1, std::memory_order_relaxed);
                live_bytes.fetch_sub(v.bytes, std::memory_order_relaxed);
                sh.map.erase(it);
            }
        }

        // Step B: relocate hot entries into the fresh write_region, then atomically swing index pointers.
        if (!reloc.empty()) {
            if (!d->reloc_buf) {
                size_t want = std::min<size_t>(kRelocChunkBytes, size_t(d->region_bytes));
                void* p = nullptr;
                if (posix_memalign(&p, kAlign, want) == 0 && p) {
                    d->reloc_buf = static_cast<uint8_t*>(p);
                    d->reloc_buf_bytes = want;
                }
            }
            size_t idx = 0;
            while (idx < reloc.size()) {
                if (!d->reloc_buf || reloc[idx].aligned > d->reloc_buf_bytes) {
                    const auto& item = reloc[idx++];
                    IndexShard& sh = shards[shard_of(item.key)];
                    std::unique_lock<std::shared_mutex> lk(sh.mu);
                    auto it = sh.map.find(item.key);
                    if (it != sh.map.end() && it->second.drive == uint8_t(d->id) &&
                        it->second.region == uint16_t(ev) && it->second.offset == item.old_offset &&
                        it->second.gen == item.old_gen) {
                        live_entries.fetch_sub(1, std::memory_order_relaxed);
                        live_bytes.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                        sh.map.erase(it);
                    }
                    continue;
                }

                size_t end = idx;
                size_t used_buf = 0;
                while (end < reloc.size() && used_buf + reloc[end].aligned <= d->reloc_buf_bytes) {
                    used_buf += reloc[end].aligned;
                    end++;
                }

                IoBatch read_batch;
                read_batch.ops.resize(end - idx);
                size_t buf_off = 0;
                for (size_t j = idx; j < end; ++j) {
                    IoOp& op = read_batch.ops[j - idx];
                    op.is_write = false;
                    op.track_region_read = false;
                    op.region = uint16_t(ev);
                    op.offset = reloc[j].old_offset;
                    op.logical_bytes = reloc[j].aligned;
                    op.aligned_bytes = reloc[j].aligned;
                    op.iovs.push_back({d->reloc_buf + buf_off, reloc[j].aligned});
                    buf_off += reloc[j].aligned;
                }
                d->enqueue_batch(&read_batch);
                read_batch.wait();

                IoBatch write_batch;
                std::vector<size_t> write_map;
                std::vector<uint64_t> new_offsets;
                buf_off = 0;
                for (size_t j = idx; j < end; ++j) {
                    size_t al = reloc[j].aligned;
                    if (read_batch.ops[j - idx].ok && d->write_pos + al <= d->region_bytes) {
                        uint64_t new_off = uint64_t(wr) * d->region_bytes + d->write_pos;
                        d->write_pos += al;
                        IoOp op;
                        op.is_write = true;
                        op.region = uint16_t(wr);
                        op.offset = new_off;
                        op.logical_bytes = al;
                        op.aligned_bytes = al;
                        op.iovs.push_back({d->reloc_buf + buf_off, al});
                        write_batch.ops.push_back(std::move(op));
                        write_map.push_back(j);
                        new_offsets.push_back(new_off);
                    }
                    buf_off += al;
                }
                if (!write_batch.ops.empty()) {
                    d->enqueue_batch(&write_batch);
                    write_batch.wait();
                }

                std::vector<bool> migrated(end - idx, false);
                for (size_t k = 0; k < write_batch.ops.size(); ++k) {
                    size_t j = write_map[k];
                    const auto& item = reloc[j];
                    if (!write_batch.ops[k].ok) continue;
                    IndexShard& sh = shards[shard_of(item.key)];
                    std::unique_lock<std::shared_mutex> lk(sh.mu);
                    auto it = sh.map.find(item.key);
                    if (it != sh.map.end() && it->second.drive == uint8_t(d->id) &&
                        it->second.region == uint16_t(ev) && it->second.offset == item.old_offset &&
                        it->second.gen == item.old_gen) {
                        it->second.region = uint16_t(wr);
                        it->second.offset = new_offsets[k];
                        it->second.gen = wr_reg.gen;
                        it->second.hot.store(0, std::memory_order_relaxed);
                        wr_reg.entries.push_back({item.key, new_offsets[k], item.bytes, wr_reg.gen});
                        d->hot_relocated.fetch_add(1, std::memory_order_relaxed);
                        migrated[j - idx] = true;
                    }
                }
                for (size_t j = idx; j < end; ++j) {
                    if (migrated[j - idx]) continue;
                    const auto& item = reloc[j];
                    IndexShard& sh = shards[shard_of(item.key)];
                    std::unique_lock<std::shared_mutex> lk(sh.mu);
                    auto it = sh.map.find(item.key);
                    if (it != sh.map.end() && it->second.drive == uint8_t(d->id) &&
                        it->second.region == uint16_t(ev) && it->second.offset == item.old_offset &&
                        it->second.gen == item.old_gen) {
                        live_entries.fetch_sub(1, std::memory_order_relaxed);
                        live_bytes.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                        sh.map.erase(it);
                    }
                }
                idx = end;
            }
        }

        // Step C (R2): Zero index entries now reference ev_reg. Wait outside DriveWorker until in_flight_reads == 0.
        if (ev_reg.in_flight_reads.load(std::memory_order_acquire) > 0) {
            d->evict_wait_count.fetch_add(1, std::memory_order_relaxed);
            std::unique_lock<std::mutex> lk(d->evict_mu);
            d->evict_cv.wait(lk, [&] { return ev_reg.in_flight_reads.load(std::memory_order_acquire) == 0; });
        }

        // Step D: Mark ev_reg clean and return it to the free pool.
        ev_reg.entries.clear();
        ev_reg.gen++;
        d->evict_region = (ev + 1) % d->num_regions;
        d->used_regions--;
        d->regions_recycled.fetch_add(1, std::memory_order_relaxed);
    }

    void flush_drive_writes(Drive* d, IoBatch& batch, std::vector<IndexKey>& keys, uint8_t* out_ok) {
        if (batch.ops.empty()) return;
        d->enqueue_batch(&batch);
        batch.wait();
        for (size_t i = 0; i < batch.ops.size(); ++i) {
            const IoOp& op = batch.ops[i];
            if (op.ok) {
                uint64_t gen = d->regions[op.region].gen;
                d->regions[op.region].entries.push_back({keys[i], op.offset, uint32_t(op.logical_bytes), gen});
                IndexShard& sh = shards[shard_of(keys[i])];
                {
                    std::unique_lock<std::shared_mutex> lk(sh.mu);
                    auto [it, inserted] = sh.map.try_emplace(
                        keys[i], uint8_t(d->id), op.region, uint32_t(op.logical_bytes), op.offset, gen);
                    if (inserted) {
                        live_entries.fetch_add(1, std::memory_order_relaxed);
                        live_bytes.fetch_add(op.logical_bytes, std::memory_order_relaxed);
                    } else {
                        live_bytes.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                        it->second.drive = uint8_t(d->id);
                        it->second.region = op.region;
                        it->second.bytes = uint32_t(op.logical_bytes);
                        it->second.offset = op.offset;
                        it->second.gen = gen;
                        it->second.hot.store(0, std::memory_order_relaxed);
                        live_bytes.fetch_add(op.logical_bytes, std::memory_order_relaxed);
                    }
                }
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 1;
                puts_ok.fetch_add(1, std::memory_order_relaxed);
                bytes_written.fetch_add(op.logical_bytes, std::memory_order_relaxed);
            } else {
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 0;
                puts_failed.fetch_add(1, std::memory_order_relaxed);
            }
        }
        batch.ops.clear();
        keys.clear();
    }
};

extern "C" {

XKV_API const char* xtier_kv_last_error(void) { return g_last_error.c_str(); }

XKV_API xtier_kv_store_t* xtier_kv_open(const char* const* paths, int num_drives, const xtier_kv_config_t* cfg) {
    g_last_error.clear();
    if (!paths || num_drives <= 0 || num_drives > 64) {
        set_error("Invalid drive list");
        return nullptr;
    }
    xtier_kv_config_t c {};
    if (cfg) c = *cfg;
    if (c.bytes_per_drive == 0) {
        set_error("bytes_per_drive must be positive");
        return nullptr;
    }
    if (c.region_bytes == 0) c.region_bytes = kDefaultRegionBytes;
    c.region_bytes = round_up_4k(c.region_bytes);
    uint64_t nreg = c.bytes_per_drive / c.region_bytes;
    if (nreg < 4 || nreg > 65535) {
        set_error("bytes_per_drive / region_bytes must be between 4 and 65535");
        return nullptr;
    }
    if (c.write_qd == 0) c.write_qd = kDefaultWriteQD;
    if (c.read_qd == 0) c.read_qd = kDefaultReadQD;
    if (c.bounce_bytes == 0) c.bounce_bytes = kDefaultBounceBytes;
    if (c.relocate_pct == 0 || c.relocate_pct > 90) c.relocate_pct = kDefaultRelocatePct;
    if (c.ring_depth == 0) c.ring_depth = kDefaultRingDepth;

    auto store = std::make_unique<xtier_kv_store>();
    store->drives.reserve(size_t(num_drives));

    for (int i = 0; i < num_drives; ++i) {
        if (!paths[i] || !paths[i][0]) {
            set_error("Empty drive path");
            return nullptr;
        }
        auto d = std::make_unique<Drive>();
        d->id = i;
        d->path = paths[i];
        d->region_bytes = c.region_bytes;
        d->num_regions = uint32_t(nreg);
        d->capacity_bytes = uint64_t(nreg) * c.region_bytes;
        d->write_qd = c.write_qd;
        d->read_qd = c.read_qd;
        d->ring_depth = c.ring_depth;
        d->relocate_pct = c.relocate_pct;
        d->pin_numa = (cfg == nullptr) ? true : ((c.flags & 1U) != 0);

        int fd = ::open(d->path.c_str(), O_RDWR | O_CREAT | O_DIRECT | O_CLOEXEC, 0664);
        if (fd < 0) {
            set_errno("open(O_DIRECT) failed", d->path, errno);
            return nullptr;
        }
        d->fd = fd;

        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            set_errno("flock(LOCK_EX|LOCK_NB) failed (pool in use by another process?)", d->path, errno);
            return nullptr;
        }

        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            set_errno("fstat failed", d->path, errno);
            return nullptr;
        }
        if (uint64_t(st.st_size) < d->capacity_bytes) {
            if (::fallocate(fd, 0, 0, off_t(d->capacity_bytes)) != 0) {
                set_errno("fallocate failed", d->path, errno);
                return nullptr;
            }
        }

        d->numa_node = detect_fd_numa_node(fd);
        d->efd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (d->efd < 0) {
            set_errno("eventfd failed", d->path, errno);
            return nullptr;
        }

        if (!d->bounce.init(c.bounce_bytes)) {
            set_error("Failed to allocate 4 KiB-aligned bounce arena");
            return nullptr;
        }

        d->regions = std::make_unique<Region[]>(d->num_regions);

        struct io_uring_params params {};
        params.flags = IORING_SETUP_COOP_TASKRUN;
        int rc = io_uring_queue_init_params(d->ring_depth, &d->ring, &params);
        if (rc < 0) {
            std::memset(&params, 0, sizeof(params));
            rc = io_uring_queue_init_params(d->ring_depth, &d->ring, &params);
        }
        if (rc < 0) {
            set_errno("io_uring_queue_init failed", d->path, -rc);
            return nullptr;
        }
        d->ring_inited = true;

        Drive* raw_d = d.get();
        d->worker = std::thread([raw_d] { raw_d->worker_loop(); });
        store->drives.push_back(std::move(d));
    }

    return store.release();
}

XKV_API void xtier_kv_close(xtier_kv_store_t* store) { delete store; }

XKV_API void xtier_kv_clear(xtier_kv_store_t* store) {
    if (!store) return;
    std::vector<std::unique_lock<std::mutex>> drive_locks;
    drive_locks.reserve(store->drives.size());
    for (auto& d : store->drives) {
        drive_locks.emplace_back(d->write_mu);
    }
    for (int s = 0; s < kNumShards; ++s) {
        std::unique_lock<std::shared_mutex> lk(store->shards[s].mu);
        store->shards[s].map.clear();
    }
    store->live_entries.store(0, std::memory_order_relaxed);
    store->live_bytes.store(0, std::memory_order_relaxed);

    for (auto& d : store->drives) {
        for (uint32_t r = 0; r < d->num_regions; ++r) {
            if (d->regions[r].in_flight_reads.load(std::memory_order_acquire) > 0) {
                std::unique_lock<std::mutex> lk(d->evict_mu);
                d->evict_cv.wait(lk, [&] {
                    return d->regions[r].in_flight_reads.load(std::memory_order_acquire) == 0;
                });
            }
            d->regions[r].entries.clear();
            d->regions[r].gen++;
        }
        d->write_region = 0;
        d->write_pos = 0;
        d->evict_region = 0;
        d->used_regions = 1;
    }
}

XKV_API void xtier_kv_stats_get(xtier_kv_store_t* store, xtier_kv_stats_t* out) {
    if (!store || !out) return;
    std::memset(out, 0, sizeof(*out));
    out->live_entries = store->live_entries.load(std::memory_order_relaxed);
    out->live_bytes = store->live_bytes.load(std::memory_order_relaxed);
    out->puts_ok = store->puts_ok.load(std::memory_order_relaxed);
    out->puts_skipped_dup = store->puts_skipped_dup.load(std::memory_order_relaxed);
    out->puts_failed = store->puts_failed.load(std::memory_order_relaxed);
    out->gets_hit = store->gets_hit.load(std::memory_order_relaxed);
    out->gets_miss = store->gets_miss.load(std::memory_order_relaxed);
    out->gets_failed = store->gets_failed.load(std::memory_order_relaxed);
    out->bytes_written = store->bytes_written.load(std::memory_order_relaxed);
    out->bytes_read = store->bytes_read.load(std::memory_order_relaxed);
    for (const auto& d : store->drives) {
        out->capacity_bytes += d->capacity_bytes;
        out->regions_recycled += d->regions_recycled.load(std::memory_order_relaxed);
        out->hot_relocated += d->hot_relocated.load(std::memory_order_relaxed);
        out->hot_dropped_cap += d->hot_dropped_cap.load(std::memory_order_relaxed);
        out->direct_ios += d->direct_ios.load(std::memory_order_relaxed);
        out->bounce_ios += d->bounce_ios.load(std::memory_order_relaxed);
        out->evict_wait_count += d->evict_wait_count.load(std::memory_order_relaxed);
    }
}

XKV_API void xtier_kv_exists(xtier_kv_store_t* store, uint16_t pool_id, const xtier_kv_key_t* keys, int count,
                             uint8_t* out_bitmap) {
    if (!store || !keys || !out_bitmap || count <= 0) return;
    std::vector<int> by_shard[kNumShards];
    for (int i = 0; i < count; ++i) {
        IndexKey k{keys[i].lo, keys[i].hi, pool_id};
        by_shard[shard_of(k)].push_back(i);
    }
    for (int s = 0; s < kNumShards; ++s) {
        if (by_shard[s].empty()) continue;
        std::shared_lock<std::shared_mutex> lk(store->shards[s].mu);
        const auto& m = store->shards[s].map;
        for (int idx : by_shard[s]) {
            IndexKey k{keys[idx].lo, keys[idx].hi, pool_id};
            out_bitmap[idx] = (m.find(k) != m.end()) ? 1 : 0;
        }
    }
}

XKV_API int xtier_kv_put(xtier_kv_store_t* store, const xtier_kv_item_t* items, int count, uint8_t* out_ok) {
    if (!store || !items || count <= 0) return 0;
    if (out_ok) std::memset(out_ok, 0, size_t(count));

    const size_t nd = store->drives.size();
    std::vector<std::vector<int>> per_drive(nd);
    std::unordered_set<IndexKey, IndexKeyHash> seen_in_batch;
    seen_in_batch.reserve(size_t(count) * 2);

    for (int i = 0; i < count; ++i) {
        const auto& it = items[i];
        IndexKey k{it.key.lo, it.key.hi, it.pool_id};
        if (!seen_in_batch.insert(k).second) {
            if (out_ok) out_ok[i] = 1;
            store->puts_skipped_dup.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        bool exists = false;
        {
            IndexShard& sh = store->shards[shard_of(k)];
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            exists = (sh.map.find(k) != sh.map.end());
        }
        if (exists) {
            if (out_ok) out_ok[i] = 1;
            store->puts_skipped_dup.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        int drv = drive_of(it.key, int(nd));
        per_drive[size_t(drv)].push_back(i);
    }

    std::vector<std::unique_lock<std::mutex>> drive_locks;
    std::vector<size_t> active_drives;
    for (size_t d = 0; d < nd; ++d) {
        if (!per_drive[d].empty()) {
            drive_locks.emplace_back(store->drives[d]->write_mu);
            active_drives.push_back(d);
        }
    }

    std::unique_ptr<IoBatch[]> batches(new IoBatch[nd]);
    std::vector<std::vector<IndexKey>> batch_keys(nd);

    for (size_t drv_idx : active_drives) {
        Drive* d = store->drives[drv_idx].get();
        auto& batch = batches[drv_idx];
        auto& bkeys = batch_keys[drv_idx];

        for (int item_idx : per_drive[drv_idx]) {
            const auto& it = items[item_idx];
            IndexKey k{it.key.lo, it.key.hi, it.pool_id};

            bool exists = false;
            {
                IndexShard& sh = store->shards[shard_of(k)];
                std::shared_lock<std::shared_mutex> lk(sh.mu);
                exists = (sh.map.find(k) != sh.map.end());
            }
            if (exists) {
                if (out_ok) out_ok[item_idx] = 1;
                store->puts_skipped_dup.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            size_t logical = 0;
            for (uint32_t j = 0; j < it.iov_count; ++j) logical += it.iovs[j].len;
            size_t aligned = round_up_4k(logical);
            if (logical == 0 || aligned > d->region_bytes) {
                if (out_ok) out_ok[item_idx] = 0;
                store->puts_failed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            if (d->write_pos + aligned > d->region_bytes) {
                if (d->used_regions + 1 > d->num_regions - 2) {
                    store->flush_drive_writes(d, batch, bkeys, out_ok);
                }
                d->write_region = (d->write_region + 1) % d->num_regions;
                d->write_pos = 0;
                d->used_regions++;
                while (d->used_regions > d->num_regions - 2) {
                    store->recycle_oldest_region(d);
                }
            }

            uint64_t off = uint64_t(d->write_region) * d->region_bytes + d->write_pos;
            d->write_pos += aligned;

            IoOp op;
            op.is_write = true;
            op.region = uint16_t(d->write_region);
            op.item_idx = item_idx;
            op.offset = off;
            op.logical_bytes = logical;
            op.aligned_bytes = aligned;
            op.iovs.reserve(it.iov_count);
            for (uint32_t j = 0; j < it.iov_count; ++j) {
                if (it.iovs[j].len > 0) op.iovs.push_back({it.iovs[j].base, it.iovs[j].len});
            }
            batch.ops.push_back(std::move(op));
            bkeys.push_back(k);
        }
    }

    for (size_t drv_idx : active_drives) {
        if (!batches[drv_idx].ops.empty()) {
            store->drives[drv_idx]->enqueue_batch(&batches[drv_idx]);
        }
    }
    for (size_t drv_idx : active_drives) {
        Drive* d = store->drives[drv_idx].get();
        auto& batch = batches[drv_idx];
        auto& bkeys = batch_keys[drv_idx];
        if (batch.ops.empty()) continue;
        batch.wait();
        for (size_t i = 0; i < batch.ops.size(); ++i) {
            const IoOp& op = batch.ops[i];
            if (op.ok) {
                uint64_t gen = d->regions[op.region].gen;
                d->regions[op.region].entries.push_back({bkeys[i], op.offset, uint32_t(op.logical_bytes), gen});
                IndexShard& sh = store->shards[shard_of(bkeys[i])];
                {
                    std::unique_lock<std::shared_mutex> lk(sh.mu);
                    auto [it, inserted] = sh.map.try_emplace(
                        bkeys[i], uint8_t(d->id), op.region, uint32_t(op.logical_bytes), op.offset, gen);
                    if (inserted) {
                        store->live_entries.fetch_add(1, std::memory_order_relaxed);
                        store->live_bytes.fetch_add(op.logical_bytes, std::memory_order_relaxed);
                    } else {
                        store->live_bytes.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                        it->second.drive = uint8_t(d->id);
                        it->second.region = op.region;
                        it->second.bytes = uint32_t(op.logical_bytes);
                        it->second.offset = op.offset;
                        it->second.gen = gen;
                        it->second.hot.store(0, std::memory_order_relaxed);
                        store->live_bytes.fetch_add(op.logical_bytes, std::memory_order_relaxed);
                    }
                }
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 1;
                store->puts_ok.fetch_add(1, std::memory_order_relaxed);
                store->bytes_written.fetch_add(op.logical_bytes, std::memory_order_relaxed);
            } else {
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 0;
                store->puts_failed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    int total_ok = 0;
    if (out_ok) {
        for (int i = 0; i < count; ++i) total_ok += out_ok[i];
    } else {
        total_ok = count;
    }
    return total_ok;
}

XKV_API int xtier_kv_get(xtier_kv_store_t* store, const xtier_kv_item_t* items, int count, uint8_t* out_ok) {
    if (!store || !items || count <= 0) return 0;
    if (out_ok) std::memset(out_ok, 0, size_t(count));

    const size_t nd = store->drives.size();
    std::unique_ptr<IoBatch[]> batches(new IoBatch[nd]);

    for (int i = 0; i < count; ++i) {
        const auto& it = items[i];
        size_t expected = 0;
        for (uint32_t j = 0; j < it.iov_count; ++j) expected += it.iovs[j].len;
        if (expected == 0) {
            store->gets_miss.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        IndexKey k{it.key.lo, it.key.hi, it.pool_id};
        IndexShard& sh = store->shards[shard_of(k)];
        bool found = false;
        uint8_t drv = 0;
        uint16_t reg = 0;
        uint64_t off = 0;
        uint32_t bytes = 0;

        {
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            auto mit = sh.map.find(k);
            if (mit != sh.map.end() && mit->second.bytes == expected) {
                IndexValue& v = mit->second;
                drv = v.drive;
                reg = v.region;
                off = v.offset;
                bytes = v.bytes;
                v.hot.store(1, std::memory_order_relaxed);
                store->drives[drv]->regions[reg].in_flight_reads.fetch_add(1, std::memory_order_acq_rel);
                found = true;
            }
        }

        if (!found) {
            store->gets_miss.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        IoOp op;
        op.is_write = false;
        op.track_region_read = true;
        op.region = reg;
        op.item_idx = i;
        op.offset = off;
        op.logical_bytes = bytes;
        op.aligned_bytes = round_up_4k(bytes);
        op.iovs.reserve(it.iov_count);
        for (uint32_t j = 0; j < it.iov_count; ++j) {
            if (it.iovs[j].len > 0) op.iovs.push_back({it.iovs[j].base, it.iovs[j].len});
        }
        batches[drv].ops.push_back(std::move(op));
    }

    uint32_t delay = store->test_read_delay_us.load(std::memory_order_relaxed);
    if (delay > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
    }

    for (size_t d = 0; d < nd; ++d) {
        if (!batches[d].ops.empty()) {
            store->drives[d]->enqueue_batch(&batches[d]);
        }
    }
    int total_ok = 0;
    for (size_t d = 0; d < nd; ++d) {
        auto& batch = batches[d];
        if (batch.ops.empty()) continue;
        batch.wait();
        for (const auto& op : batch.ops) {
            if (op.ok) {
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 1;
                total_ok++;
                store->gets_hit.fetch_add(1, std::memory_order_relaxed);
                store->bytes_read.fetch_add(op.logical_bytes, std::memory_order_relaxed);
            } else {
                if (out_ok && op.item_idx >= 0) out_ok[op.item_idx] = 0;
                store->gets_failed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return total_ok;
}

XKV_API void xtier_kv_set_test_read_delay_us(xtier_kv_store_t* store, uint32_t delay_us) {
    if (store) store->test_read_delay_us.store(delay_us, std::memory_order_relaxed);
}

}  // extern "C"
