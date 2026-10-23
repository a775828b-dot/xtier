// XTier row tables: exact rows of large embedding tables read straight from the checkpoint files on NVMe, for lookups
// that run as cudaLaunchHostFunc nodes of captured CUDA graphs (this file calls no CUDA API).
//
// Grown out of the Engram row store of the DeepSeek V4.1 1017 pipeline (adapter/row_store.cpp, 2026-10-22): rows of
// any width with an optional block of scale bytes per row (DeepSeek V4.1 Engram: 256 FP8 bytes + 8 E8M0 scales;
// Qwen3.8 PLE n-gram table: 160 FP8 bytes, no scales), one table split over segments (checkpoint tensors) in one or
// more files. Misses are read with O_DIRECT by a pool of reader threads (XTIER_ROWS_IO_THREADS, default 16) in rounds
// of up to kBatchPages 4 KiB pages; a bounded direct-mapped cache keeps recent rows; the scales of the owned rows can
// be preloaded. XTIER_ROWS_MODE=ram maps and locks the files instead (no cache, no reads). A lookup returns zero bytes
// for ids outside the owned range [lo, hi) (TP shards: the all-reduce adds them up) and aborts the process on ids
// outside the table or on failed reads: a generation never continues with wrong rows.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#define XTIER_API extern "C" __attribute__((visibility("default")))

static constexpr size_t kPage = 4096;
// Misses of one lookup share read rounds of up to kBatchPages pages (2026-10-19: one round per 32 rows cost ~40K
// serial rounds per layer for a fresh 93K-token prefill).
static constexpr size_t kBatchPages = 512;
static constexpr size_t kBounce = 8u << 20;   // read size of the scale preload

struct Segment {
  int file;                               // index into Store::fds
  uint64_t first, rows;                   // global rows [first, first + rows)
  uint64_t weight_offset, scale_offset;   // file offsets of the segment's row 0
};

struct Store {
  std::vector<std::string> paths;
  std::vector<int> fds;
  std::vector<size_t> sizes;
  std::vector<uint8_t *> maps;            // XTIER_ROWS_MODE=ram: whole-file mappings
  std::vector<Segment> segments;
  uint64_t rows = 0, row_bytes = 0, scale_bytes = 0, entry = 0;
  uint64_t uniform = 0;                   // rows of every segment but the last when they are all equal, else 0
  uint64_t row_lo = 0, row_hi = 0;
  bool ram = false;
  uint8_t *pages = nullptr;               // kBatchPages aligned read buffers
  uint8_t *scale_data = nullptr;          // preloaded scales of rows [scale_row_lo, scale_row_hi)
  size_t scale_allocation = 0;
  uint64_t scale_row_lo = 0, scale_row_hi = 0;
  uint64_t slots = 0;
  uint8_t *cache = nullptr;               // slots x entry bytes: the row, then its scales
  uint64_t *keys = nullptr;               // id + 1 of the row in each slot, 0 = empty
  std::mutex batch_lock;                  // one lookup at a time owns the read buffers and the cache
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0}, batches{0}, peak_pages{0}, scale_loaded{0};
  std::atomic<uint64_t> lookup_ns{0}, lookups{0}, lookup_max_ns{0};
};

struct Work {   // one lookup, the argument of the host function
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;   // count x row_bytes, count x scale_bytes
  uint64_t count;
};

[[noreturn]] static void fail(const char *reason) {
  std::fprintf(stderr, "xtier row table: %s (errno=%d)\n", reason, errno);
  std::abort();   // never let a generation continue with missing or stale rows
}

static const Segment &segment_of(const Store *s, uint64_t id) {
  const size_t n = s->segments.size();
  if (s->uniform) return s->segments[std::min<uint64_t>(id / s->uniform, n - 1)];
  size_t lo = 0, hi = n;   // the last segment that starts at or before id
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    (s->segments[mid].first <= id ? lo : hi) = mid;
  }
  return s->segments[lo];
}

static uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

// Persistent reader threads (XTIER_ROWS_IO_THREADS, default 16): the O_DIRECT reads of one round run concurrently, so
// the drive sees real queue depth (native AIO completed them one by one here). The calling (CUDA callback) thread
// takes part as well. One round at a time.
struct IoPool {
  std::mutex job_lock, m;
  std::condition_variable start, finish;
  std::vector<std::thread> threads;
  uint64_t generation = 0;
  const int *fds = nullptr;
  const uint64_t *offsets = nullptr;
  const uint32_t *required = nullptr;
  uint8_t *pages = nullptr;
  size_t count = 0, active = 0;
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};

  void run_share() {
    for (size_t j; (j = next.fetch_add(1)) < count;) {
      ssize_t got;
      do { got = pread(fds[j], pages + j * kPage, kPage, offsets[j]); } while (got < 0 && errno == EINTR);
      if (got < 0 || uint64_t(got) < required[j]) failed = true;
    }
  }
  void worker() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m);
        start.wait(lock, [&] { return generation != seen; });
        seen = generation;
      }
      run_share();
      std::lock_guard<std::mutex> lock(m);
      if (--active == 0) finish.notify_one();
    }
  }
  IoPool() {
    const char *env = std::getenv("XTIER_ROWS_IO_THREADS");
    const int n = env ? std::atoi(env) : 16;
    for (int k = 0; k < std::max(0, n); ++k) threads.emplace_back([this] { worker(); });
    for (auto &t : threads) t.detach();
  }
  void read(const int *fds_, const uint64_t *offsets_, const uint32_t *required_, uint8_t *pages_, size_t count_) {
    std::lock_guard<std::mutex> job(job_lock);
    {
      std::lock_guard<std::mutex> lock(m);
      fds = fds_; offsets = offsets_; required = required_; pages = pages_; count = count_;
      next = 0; failed = false; active = threads.size(); ++generation;
    }
    start.notify_all();
    run_share();
    std::unique_lock<std::mutex> lock(m);
    finish.wait(lock, [&] { return active == 0; });
    if (failed) fail("short or failed direct read");
  }
};

static IoPool &io_pool() {
  static IoPool *pool = new IoPool();   // never destroyed: the threads are detached
  return *pool;
}

// Copy `length` bytes at file offset `offset` out of the consecutive read buffers from `page` on.
static void copy_pages(const Store *s, uint32_t page, uint64_t offset, uint8_t *out, size_t length) {
  size_t delta = offset & (kPage - 1);
  while (length) {
    const size_t take = std::min(length, kPage - delta);
    std::memcpy(out, s->pages + size_t(page) * kPage + delta, take);
    out += take; length -= take; delta = 0; ++page;
  }
}

struct Miss { uint64_t index, id, weight_offset, scale_offset; uint32_t weight_page, scale_page; };

static void lookup_direct(Work *work) {
  Store *s = work->store;
  std::lock_guard<std::mutex> guard(s->batch_lock);
  static thread_local std::vector<int> fds(kBatchPages);
  static thread_local std::vector<uint64_t> offsets(kBatchPages);
  static thread_local std::vector<uint32_t> required(kBatchPages);
  static thread_local std::vector<Miss> misses(kBatchPages);
  const uint64_t W = s->row_bytes, SB = s->scale_bytes;
  const bool read_scales = SB && !s->scale_data;
  // pages one miss can need: n bytes touch at most n / 4096 + 2 pages
  const size_t need = W / kPage + 2 + (read_scales ? SB / kPage + 2 : 0);
  if (need > kBatchPages) fail("row wider than a read round");
  size_t pages = 0, miss_count = 0;
  auto add_pages = [&](int fd, uint64_t offset, size_t length) {
    const uint32_t first = uint32_t(pages);
    uint64_t base = offset & ~uint64_t(kPage - 1);
    size_t left = (offset - base) + length;
    while (left) {
      const size_t take = std::min(left, kPage);
      fds[pages] = fd; offsets[pages] = base; required[pages] = uint32_t(take);
      ++pages; base += kPage; left -= take;
    }
    return first;
  };
  auto flush = [&] {
    if (pages) {
      io_pool().read(fds.data(), offsets.data(), required.data(), s->pages, pages);
      s->reads.fetch_add(pages, std::memory_order_relaxed);
      s->batches.fetch_add(1, std::memory_order_relaxed);
      if (pages > s->peak_pages.load(std::memory_order_relaxed)) s->peak_pages.store(pages, std::memory_order_relaxed);
    }
    for (size_t k = 0; k < miss_count; ++k) {
      const Miss &m = misses[k];
      uint8_t *row = work->weights + m.index * W, *scales = SB ? work->scales + m.index * SB : nullptr;
      copy_pages(s, m.weight_page, m.weight_offset, row, W);
      if (SB) {
        if (s->scale_data) std::memcpy(scales, s->scale_data + (m.id - s->scale_row_lo) * SB, SB);
        else copy_pages(s, m.scale_page, m.scale_offset, scales, SB);
      }
      if (s->slots) {
        const uint64_t slot = m.id % s->slots;
        std::memcpy(s->cache + slot * s->entry, row, W);
        if (SB) std::memcpy(s->cache + slot * s->entry + W, scales, SB);
        s->keys[slot] = m.id + 1;
      }
    }
    pages = miss_count = 0;
  };
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row id out of bounds");
    uint8_t *row = work->weights + i * W;
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(row, 0, W);
      if (SB) std::memset(work->scales + i * SB, 0, SB);
      continue;
    }
    if (s->slots) {
      const uint64_t slot = uint64_t(id) % s->slots;
      if (s->keys[slot] == uint64_t(id) + 1) {
        std::memcpy(row, s->cache + slot * s->entry, W);
        if (SB) std::memcpy(work->scales + i * SB, s->cache + slot * s->entry + W, SB);
        s->hits.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    }
    if (pages + need > kBatchPages || miss_count == kBatchPages) flush();
    const Segment &g = segment_of(s, uint64_t(id));
    const uint64_t r = uint64_t(id) - g.first;
    Miss m{i, uint64_t(id), g.weight_offset + r * W, g.scale_offset + r * SB, 0, 0};
    m.weight_page = add_pages(s->fds[g.file], m.weight_offset, W);
    if (read_scales) m.scale_page = add_pages(s->fds[g.file], m.scale_offset, SB);
    misses[miss_count++] = m;
    s->misses.fetch_add(1, std::memory_order_relaxed);
  }
  flush();
}

static void lookup_resident(Work *work) {
  const Store *s = work->store;
  const uint64_t W = s->row_bytes, SB = s->scale_bytes;
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row id out of bounds");
    uint8_t *row = work->weights + i * W, *scales = SB ? work->scales + i * SB : nullptr;
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(row, 0, W);
      if (SB) std::memset(scales, 0, SB);
      continue;
    }
    const Segment &g = segment_of(s, uint64_t(id));
    const uint64_t r = uint64_t(id) - g.first;
    const uint8_t *map = s->maps[g.file];
    std::memcpy(row, map + g.weight_offset + r * W, W);
    if (SB) std::memcpy(scales, map + g.scale_offset + r * SB, SB);
  }
}

static void destroy(Store *s) {
  for (size_t k = 0; k < s->fds.size(); ++k) {
    if (s->maps[k]) munmap(s->maps[k], s->sizes[k]);
    close(s->fds[k]);
  }
  if (s->cache) munmap(s->cache, s->slots * s->entry);
  if (s->keys) munmap(s->keys, s->slots * sizeof(uint64_t));
  if (s->scale_data) munmap(s->scale_data, s->scale_allocation);
  std::free(s->pages);
  delete s;
}

static int open_file(Store *s, const char *path) {
  for (size_t k = 0; k < s->paths.size(); ++k)
    if (s->paths[k] == path) return int(k);
  const int fd = open(path, O_RDONLY | O_CLOEXEC | (s->ram ? 0 : O_DIRECT));
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st)) {
    const int error = errno;
    close(fd);
    errno = error;
    return -1;
  }
  s->paths.emplace_back(path);
  s->fds.push_back(fd);
  s->sizes.push_back(size_t(st.st_size));
  s->maps.push_back(nullptr);
  return int(s->fds.size() - 1);
}

static Store *give_up(Store *s, int error) {
  destroy(s);
  errno = error;
  return nullptr;
}

// One table of `nseg` segments in global row order: segment k has seg_rows[k] rows of row_bytes bytes at
// weight_offsets[k] of paths[k] and, when scale_bytes > 0, their scale bytes at scale_offsets[k]. cache_bytes bounds
// the row cache (entries plus keys). Returns nullptr with errno on failure.
XTIER_API Store *xtier_rows_open(uint64_t nseg, const char *const *paths, const uint64_t *seg_rows,
                                 const uint64_t *weight_offsets, const uint64_t *scale_offsets, uint64_t row_bytes,
                                 uint64_t scale_bytes, uint64_t cache_bytes) {
  if (!nseg || !row_bytes) {
    errno = EINVAL;
    return nullptr;
  }
  auto *s = new Store;
  const char *mode = std::getenv("XTIER_ROWS_MODE");
  s->ram = mode && std::strcmp(mode, "ram") == 0;
  s->row_bytes = row_bytes; s->scale_bytes = scale_bytes; s->entry = row_bytes + scale_bytes;
  for (uint64_t k = 0; k < nseg; ++k) {
    const int file = open_file(s, paths[k]);
    if (file < 0) return give_up(s, errno);
    const uint64_t size = s->sizes[file], n = seg_rows[k];
    const bool fits = weight_offsets[k] <= size && n <= (size - weight_offsets[k]) / row_bytes &&
                      (!scale_bytes || (scale_offsets[k] <= size && n <= (size - scale_offsets[k]) / scale_bytes));
    if (!fits) return give_up(s, ERANGE);
    s->segments.push_back({file, s->rows, n, weight_offsets[k], scale_bytes ? scale_offsets[k] : 0});
    s->rows += n;
  }
  s->uniform = s->segments[0].rows;
  for (size_t k = 0; k + 1 < s->segments.size(); ++k)
    if (s->segments[k].rows != s->uniform) s->uniform = 0;
  s->row_hi = s->rows;
  if (s->ram) {
    for (size_t k = 0; k < s->fds.size(); ++k) {
      void *map = mmap(nullptr, s->sizes[k], PROT_READ, MAP_SHARED | MAP_POPULATE, s->fds[k], 0);
      if (map == MAP_FAILED) return give_up(s, errno);
      s->maps[k] = static_cast<uint8_t *>(map);
      if (mlock(map, s->sizes[k])) return give_up(s, errno);   // needs the memlock limit and the RAM
    }
    return s;
  }
  if (posix_memalign(reinterpret_cast<void **>(&s->pages), kPage, kBatchPages * kPage)) return give_up(s, ENOMEM);
  s->slots = cache_bytes / (s->entry + sizeof(uint64_t));
  if (s->slots) {
    void *cache = mmap(nullptr, s->slots * s->entry, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *keys = mmap(nullptr, s->slots * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    s->cache = cache == MAP_FAILED ? nullptr : static_cast<uint8_t *>(cache);
    s->keys = keys == MAP_FAILED ? nullptr : static_cast<uint64_t *>(keys);
    if (!s->cache || !s->keys) return give_up(s, ENOMEM);
  }
  return s;
}

// Time spent inside the host function (cache, reads, copies); the GPU-visible gap minus this is the dispatch latency.
XTIER_API void xtier_rows_lookup(void *opaque) {
  const uint64_t start = now_ns();
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  if (s->ram) lookup_resident(work);
  else lookup_direct(work);
  const uint64_t took = now_ns() - start;
  s->lookup_ns.fetch_add(took, std::memory_order_relaxed);
  s->lookups.fetch_add(1, std::memory_order_relaxed);
  uint64_t seen = s->lookup_max_ns.load(std::memory_order_relaxed);
  while (took > seen && !s->lookup_max_ns.compare_exchange_weak(seen, took)) {}
}

// Reads the scales of the owned rows [row_lo, row_hi) into memory (the whole table on one GPU; a TP rank never looks
// up other rows), so a miss reads only its row. Returns 0, or -1 with errno.
XTIER_API int xtier_rows_preload_scales(Store *s) {
  std::lock_guard<std::mutex> guard(s->batch_lock);
  const uint64_t SB = s->scale_bytes, lo = s->row_lo, hi = s->row_hi;
  if (!SB || s->scale_data || s->ram) return 0;
  const size_t allocation = std::max(((hi - lo) * SB + kPage - 1) & ~(kPage - 1), kPage);
  void *memory = mmap(nullptr, allocation, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED) return -1;
  auto *data = static_cast<uint8_t *>(memory);
  uint8_t *bounce = nullptr;
  if (posix_memalign(reinterpret_cast<void **>(&bounce), kPage, kBounce)) {
    munmap(memory, allocation);
    errno = ENOMEM;
    return -1;
  }
  int error = 0;
  for (const Segment &g : s->segments) {
    const uint64_t a = std::max(lo, g.first), b = std::min(hi, g.first + g.rows);
    uint64_t offset = g.scale_offset + (a - g.first) * SB;
    size_t left = a < b ? (b - a) * SB : 0;
    uint8_t *out = data + (a - lo) * SB;
    while (left && !error) {
      const uint64_t base = offset & ~uint64_t(kPage - 1);
      const size_t delta = offset - base, take = std::min(left, kBounce - delta);
      const size_t request = (delta + take + kPage - 1) & ~(kPage - 1);
      ssize_t got;
      do { got = pread(s->fds[g.file], bounce, request, base); } while (got < 0 && errno == EINTR);
      if (got < 0 || size_t(got) < delta + take) {
        error = got < 0 ? errno : EIO;
        break;
      }
      std::memcpy(out, bounce + delta, take);
      out += take; offset += take; left -= take;
    }
    if (error) break;
  }
  std::free(bounce);
  if (error) {
    munmap(memory, allocation);
    errno = error;
    return -1;
  }
  s->scale_data = data; s->scale_allocation = allocation;
  s->scale_row_lo = lo; s->scale_row_hi = hi;
  s->scale_loaded.store((hi - lo) * SB);
  return 0;
}

XTIER_API void xtier_rows_range(Store *s, uint64_t lo, uint64_t hi) {
  std::lock_guard<std::mutex> guard(s->batch_lock);
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  if (s->scale_data && (lo < s->scale_row_lo || hi > s->scale_row_hi))
    fail("row ownership range outside the preloaded scales");
  s->row_lo = lo; s->row_hi = hi;
}

XTIER_API void xtier_rows_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load(); out[1] = s->misses.load(); out[2] = s->reads.load();
  out[3] = s->slots * (s->entry + sizeof(uint64_t));
}

XTIER_API void xtier_rows_io_stats(Store *s, uint64_t *out) {
  out[0] = s->batches.load(); out[1] = s->peak_pages.load(); out[2] = s->scale_loaded.load();
}

XTIER_API void xtier_rows_timing(Store *s, uint64_t *out) {
  out[0] = s->lookup_ns.load(); out[1] = s->lookups.load(); out[2] = s->lookup_max_ns.load();
}

// No lookup may be running or queued on a stream.
XTIER_API void xtier_rows_close(Store *s) { destroy(s); }
