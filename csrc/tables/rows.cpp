// XTier row tables: exact rows of large embedding tables read straight from the checkpoint files on NVMe, for lookups
// that run as cudaLaunchHostFunc nodes of captured CUDA graphs (this file calls no CUDA API).
//
// Grown out of the Engram row store of the DeepSeek V4.1 1017 pipeline (adapter/row_store.cpp, 2026-10-22): rows of
// any width with an optional block of scale bytes per row (DeepSeek V4.1 Engram: 256 FP8 bytes + 8 E8M0 scales;
// Qwen3.8 PLE n-gram table: 160 FP8 bytes, no scales), one table split over segments (checkpoint tensors) in one or
// more files. Misses are deduplicated by 4 KiB page, sorted in ascending LBA order, and read with O_DIRECT by a pool
// of reader threads (XTIER_ROWS_IO_THREADS, default 32) in rounds of up to kBatchPages 4 KiB pages; a bounded
// direct-mapped cache keeps recent rows; the scales of the owned rows can be preloaded. XTIER_ROWS_MODE=ram maps and
// locks the files instead (no cache, no reads). A lookup returns zero bytes for ids outside the owned range [lo, hi)
// (TP shards: the all-reduce adds them up) and aborts the process on ids outside the table or on failed reads: a
// generation never continues with wrong rows.
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
#include <sys/syscall.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define XTIER_API extern "C" __attribute__((visibility("default")))

static constexpr size_t kPage = 4096;
static constexpr size_t kLegacyBatchPages = 512;
static constexpr size_t kBatchPages = 16384;   // 64 MiB staging per store: avoids ~500 barriers on 49K-token chunks
static constexpr size_t kBounce = 8u << 20;    // read size of the scale preload

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

// Interleave a fresh anonymous mapping over the NUMA nodes with memory (best effort, like xtier/engine/memory.py for
// the pinned tables): the cache and the scales (DeepSeek: ~10 GB) would otherwise land on the node of the thread
// that touches them first, beside the engine's node-bound expert shards, which then may find no pages there.
static void interleave(void *addr, size_t length) {
  static const unsigned long mask = [] {
    unsigned long m = 0;
    if (FILE *f = std::fopen("/sys/devices/system/node/has_memory", "r")) {
      char text[256] = {0};
      const size_t n = std::fread(text, 1, sizeof(text) - 1, f);
      std::fclose(f);
      text[n] = 0;
      for (char *p = text; *p && *p != '\n';) {   // "0-7" or "0,2-3"
        char *end;
        const long lo = std::strtol(p, &end, 10);
        if (end == p) break;
        long hi = lo;
        if (*end == '-') hi = std::strtol(end + 1, &end, 10);
        for (long k = lo; k <= hi && k < 64; ++k) m |= 1ul << k;
        p = *end == ',' ? end + 1 : end;
      }
    }
    return m;
  }();
  if (mask) syscall(SYS_mbind, addr, length, 3 /* MPOL_INTERLEAVE */, &mask, 65ul, 0u);
}

static uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

// Persistent reader threads (XTIER_ROWS_IO_THREADS, default 32): the O_DIRECT reads of one round run concurrently, so
// the drive sees real queue depth. The calling (CUDA callback) thread takes part as well. One round at a time.
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
    for (size_t j; (j = next.fetch_add(1, std::memory_order_relaxed)) < count;) {
      ssize_t got;
      do { got = pread(fds[j], pages + j * kPage, kPage, offsets[j]); } while (got < 0 && errno == EINTR);
      if (got < 0 || uint64_t(got) < required[j]) failed.store(true, std::memory_order_relaxed);
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
    const int n = env ? std::atoi(env) : 32;
    for (int k = 0; k < std::max(0, n); ++k) threads.emplace_back([this] { worker(); });
    for (auto &t : threads) t.detach();
  }
  void read(const int *fds_, const uint64_t *offsets_, const uint32_t *required_, uint8_t *pages_, size_t count_) {
    if (count_ == 0) return;
    if (count_ == 1 || threads.empty()) {
      for (size_t j = 0; j < count_; ++j) {
        ssize_t got;
        do { got = pread(fds_[j], pages_ + j * kPage, kPage, offsets_[j]); } while (got < 0 && errno == EINTR);
        if (got < 0 || uint64_t(got) < required_[j]) fail("short or failed direct read");
      }
      return;
    }
    std::lock_guard<std::mutex> job(job_lock);
    {
      std::lock_guard<std::mutex> lock(m);
      fds = fds_; offsets = offsets_; required = required_; pages = pages_; count = count_;
      next.store(0, std::memory_order_relaxed);
      failed.store(false, std::memory_order_relaxed);
      active = threads.size();
      ++generation;
    }
    start.notify_all();
    run_share();
    std::unique_lock<std::mutex> lock(m);
    finish.wait(lock, [&] { return active == 0; });
    if (failed.load(std::memory_order_relaxed)) fail("short or failed direct read");
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

struct LegacyMiss { uint64_t index, id, weight_offset, scale_offset; uint32_t weight_page, scale_page; };

static void lookup_direct_legacy(Work *work) {
  Store *s = work->store;
  static thread_local std::vector<int> fds(kLegacyBatchPages);
  static thread_local std::vector<uint64_t> offsets(kLegacyBatchPages);
  static thread_local std::vector<uint32_t> required(kLegacyBatchPages);
  static thread_local std::vector<LegacyMiss> misses(kLegacyBatchPages);
  const uint64_t W = s->row_bytes, SB = s->scale_bytes;
  const bool read_scales = SB && !s->scale_data;
  const size_t need = W / kPage + 2 + (read_scales ? SB / kPage + 2 : 0);
  if (need > kLegacyBatchPages) fail("row wider than a read round");
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
      const LegacyMiss &m = misses[k];
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
    if (pages + need > kLegacyBatchPages || miss_count == kLegacyBatchPages) flush();
    const Segment &g = segment_of(s, uint64_t(id));
    const uint64_t r = uint64_t(id) - g.first;
    LegacyMiss m{i, uint64_t(id), g.weight_offset + r * W, g.scale_offset + r * SB, 0, 0};
    m.weight_page = add_pages(s->fds[g.file], m.weight_offset, W);
    if (read_scales) m.scale_page = add_pages(s->fds[g.file], m.scale_offset, SB);
    misses[miss_count++] = m;
    s->misses.fetch_add(1, std::memory_order_relaxed);
  }
  flush();
}

struct MissItem {
  uint64_t out_idx;
  uint64_t id;
  uint64_t w_off;
  uint64_t s_off;
  int file;
};

struct DupItem {
  uint64_t dst_idx;
  uint64_t src_idx;
};

struct PagePiece {
  uint64_t page_key;   // (uint64_t(file) << 52) | (base >> 12)
  uint8_t *dst;
  uint16_t delta;
  uint16_t take;
  bool operator<(const PagePiece &o) const { return page_key < o.page_key; }
};

struct SeenTable {
  std::vector<uint64_t> keys;
  std::vector<uint64_t> vals;
  std::vector<uint32_t> gens;
  uint32_t gen = 1;

  void reset(size_t max_items) {
    size_t cap = 256;
    while (cap < max_items * 2) cap <<= 1;
    if (keys.size() < cap) {
      keys.assign(cap, 0);
      vals.assign(cap, 0);
      gens.assign(cap, 0);
      gen = 1;
    } else if (++gen == 0) {
      std::fill(gens.begin(), gens.end(), 0);
      gen = 1;
    }
  }

  bool find_or_insert(uint64_t key, uint64_t val_new, uint64_t *existing) {
    const size_t mask = keys.size() - 1;
    size_t h = (key * 11400714819323198485ull) & mask;
    for (;;) {
      if (gens[h] != gen) {
        gens[h] = gen;
        keys[h] = key;
        vals[h] = val_new;
        return false;
      }
      if (keys[h] == key) {
        if (existing) *existing = vals[h];
        return true;
      }
      h = (h + 1) & mask;
    }
  }
};

static void emit_pieces_into(std::vector<PagePiece> &pieces, int file, uint64_t offset, size_t length, uint8_t *dst) {
  uint64_t cur_off = offset;
  size_t left = length;
  uint8_t *cur_dst = dst;
  while (left) {
    const uint64_t base = cur_off & ~uint64_t(kPage - 1);
    const size_t delta = size_t(cur_off - base);
    const size_t take = std::min(left, kPage - delta);
    const uint64_t key = (uint64_t(uint32_t(file)) << 52) | (base >> 12);
    pieces.push_back({key, cur_dst, uint16_t(delta), uint16_t(take)});
    cur_off += take;
    cur_dst += take;
    left -= take;
  }
}

static void read_sorted_pieces(Store *s, std::vector<PagePiece> &pieces) {
  if (pieces.empty()) return;
  static thread_local std::vector<int> u_fds;
  static thread_local std::vector<uint64_t> u_offsets;
  static thread_local std::vector<uint32_t> u_required;
  static thread_local std::vector<size_t> u_bounds;

  std::sort(pieces.begin(), pieces.end());

  u_fds.clear();
  u_offsets.clear();
  u_required.clear();
  u_bounds.clear();
  const size_t np = pieces.size();
  u_bounds.push_back(0);
  for (size_t p = 0; p < np;) {
    const uint64_t key = pieces[p].page_key;
    const int file = int(key >> 52);
    const uint64_t base = (key & ((1ull << 52) - 1)) << 12;
    uint32_t req = 0;
    size_t q = p;
    while (q < np && pieces[q].page_key == key) {
      const uint32_t end = uint32_t(pieces[q].delta) + uint32_t(pieces[q].take);
      if (end > req) req = end;
      ++q;
    }
    u_fds.push_back(s->fds[file]);
    u_offsets.push_back(base);
    u_required.push_back(req);
    u_bounds.push_back(q);
    p = q;
  }

  const size_t total_pages = u_fds.size();
  for (size_t u0 = 0; u0 < total_pages; u0 += kBatchPages) {
    const size_t batch_n = std::min(kBatchPages, total_pages - u0);
    io_pool().read(u_fds.data() + u0, u_offsets.data() + u0, u_required.data() + u0, s->pages, batch_n);
    s->reads.fetch_add(batch_n, std::memory_order_relaxed);
    s->batches.fetch_add(1, std::memory_order_relaxed);
    if (batch_n > s->peak_pages.load(std::memory_order_relaxed))
      s->peak_pages.store(batch_n, std::memory_order_relaxed);
    for (size_t b = 0; b < batch_n; ++b) {
      const uint8_t *page_ptr = s->pages + b * kPage;
      const size_t p0 = u_bounds[u0 + b], p1 = u_bounds[u0 + b + 1];
      for (size_t p = p0; p < p1; ++p) {
        std::memcpy(pieces[p].dst, page_ptr + pieces[p].delta, pieces[p].take);
      }
    }
  }
}

static void lookup_direct(Work *work) {
  Store *s = work->store;
  std::lock_guard<std::mutex> guard(s->batch_lock);
  const char *leg = std::getenv("XTIER_ROWS_LEGACY");
  if (leg && std::strcmp(leg, "1") == 0) {
    lookup_direct_legacy(work);
    return;
  }
  const uint64_t W = s->row_bytes, SB = s->scale_bytes;
  const bool read_scales = SB && !s->scale_data;
  static thread_local std::vector<MissItem> miss_list;
  static thread_local std::vector<DupItem> dup_list;
  static thread_local std::vector<PagePiece> pieces;
  static thread_local SeenTable seen;

  miss_list.clear();
  dup_list.clear();
  seen.reset(work->count);
  uint64_t hits_local = 0;
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
        ++hits_local;
        continue;
      }
    }
    uint64_t first_idx = 0;
    if (seen.find_or_insert(uint64_t(id) + 1, i, &first_idx)) {
      dup_list.push_back({i, first_idx});
      ++hits_local;
      continue;
    }
    const Segment &g = segment_of(s, uint64_t(id));
    if (g.file >= 4096) fail("too many segment files for page key");
    const uint64_t r = uint64_t(id) - g.first;
    miss_list.push_back({i, uint64_t(id), g.weight_offset + r * W, g.scale_offset + r * SB, g.file});
  }
  if (hits_local) s->hits.fetch_add(hits_local, std::memory_order_relaxed);
  if (miss_list.empty()) return;
  s->misses.fetch_add(miss_list.size(), std::memory_order_relaxed);

  pieces.clear();
  pieces.reserve(miss_list.size() * (read_scales ? 2 : 1) + 16);
  for (const MissItem &m : miss_list) {
    emit_pieces_into(pieces, m.file, m.w_off, W, work->weights + m.out_idx * W);
    if (SB) {
      uint8_t *scales = work->scales + m.out_idx * SB;
      if (s->scale_data) std::memcpy(scales, s->scale_data + (m.id - s->scale_row_lo) * SB, SB);
      else emit_pieces_into(pieces, m.file, m.s_off, SB, scales);
    }
  }

  read_sorted_pieces(s, pieces);

  for (const DupItem &d : dup_list) {
    std::memcpy(work->weights + d.dst_idx * W, work->weights + d.src_idx * W, W);
    if (SB) std::memcpy(work->scales + d.dst_idx * SB, work->scales + d.src_idx * SB, SB);
  }

  if (s->slots) {
    for (const MissItem &m : miss_list) {
      const uint64_t slot = m.id % s->slots;
      const uint8_t *row = work->weights + m.out_idx * W;
      std::memcpy(s->cache + slot * s->entry, row, W);
      if (SB) std::memcpy(s->cache + slot * s->entry + W, work->scales + m.out_idx * SB, SB);
      s->keys[slot] = m.id + 1;
    }
  }
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
    interleave(s->cache, s->slots * s->entry);
    interleave(s->keys, s->slots * sizeof(uint64_t));
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

// Warm `s->cache` for `ids[0..count)` on a background thread without output buffers (cross-chunk Engram lookahead).
XTIER_API void xtier_rows_prefetch(Store *s, const int64_t *ids, uint64_t count) {
  if (!s || s->ram || !s->slots || count == 0) return;
  std::lock_guard<std::mutex> guard(s->batch_lock);
  const uint64_t W = s->row_bytes, SB = s->scale_bytes;
  static thread_local SeenTable seen_slots;
  static thread_local std::vector<std::pair<uint64_t, uint64_t>> fill_slots;   // (slot, id)
  static thread_local std::vector<PagePiece> pieces;

  seen_slots.reset(count);
  fill_slots.clear();
  pieces.clear();
  for (uint64_t rev = count; rev > 0; --rev) {
    const int64_t id = ids[rev - 1];
    if (id < 0 || uint64_t(id) >= s->rows) continue;
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) continue;
    const uint64_t slot = uint64_t(id) % s->slots;
    if (seen_slots.find_or_insert(slot + 1, uint64_t(id), nullptr)) continue;
    if (s->keys[slot] == uint64_t(id) + 1) continue;
    const Segment &g = segment_of(s, uint64_t(id));
    if (g.file >= 4096) continue;
    const uint64_t r = uint64_t(id) - g.first;
    s->keys[slot] = 0;   // invalidate until read completes
    uint8_t *dst = s->cache + slot * s->entry;
    emit_pieces_into(pieces, g.file, g.weight_offset + r * W, W, dst);
    if (SB) {
      if (s->scale_data) std::memcpy(dst + W, s->scale_data + (uint64_t(id) - s->scale_row_lo) * SB, SB);
      else emit_pieces_into(pieces, g.file, g.scale_offset + r * SB, SB, dst + W);
    }
    fill_slots.emplace_back(slot, uint64_t(id));
  }
  if (fill_slots.empty()) return;
  read_sorted_pieces(s, pieces);
  for (const auto &kv : fill_slots) {
    s->keys[kv.first] = kv.second + 1;
  }
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
  interleave(memory, allocation);
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
