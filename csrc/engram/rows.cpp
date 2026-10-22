// Ported from the DeepSeek V4.1 1017 pipeline (adapter/row_store.cpp, 2026-10-22) by tools/port_engram.py:
// exported names and environment switches renamed only.
// Exact immutable FP8 row retrieval; bounded direct-mapped RAM cache.
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
#include <algorithm>
#include <atomic>
#include <array>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <time.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Store {
  int fd;
  aio_context_t aio = 0;
  uint8_t *pages = nullptr;
  uint8_t *scale_storage = nullptr, *scale_data = nullptr;   // scale_data[0] = row scale_row_lo
  size_t scale_allocation = 0;
  uint64_t scale_row_lo = 0, scale_row_hi = 0;
  std::atomic<uint64_t> scale_bytes{0};
  std::mutex batch_lock;
  std::atomic<uint64_t> batches{0}, peak_pages{0};
  uint64_t rows, weight_offset, scale_offset, slots, row_lo, row_hi;
  uint8_t *cache;
  uint64_t *keys;
  uint8_t *resident = nullptr;
  size_t resident_size = 0;
  std::mutex locks[256];
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0};
  std::atomic<uint64_t> lookup_ns{0}, lookups{0}, lookup_max_ns{0};
};
// 2026-10-19: misses of a whole lookup share AIO rounds of up to kBatchPages page reads (the
// old code paid one round per 32 rows: ~40K serial rounds per layer for a fresh 93K-token prefill).
static constexpr size_t kBatchPages = 512;

struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

static void fail(const char *reason) {
  std::fprintf(stderr, "Engram retrieval failed: %s (errno=%d)\n", reason, errno);
  std::abort(); // Never allow a generation to continue with missing/stale rows.
}

static void read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident) {
    std::memcpy(out, s->resident + offset, length);
    return;
  }
  alignas(4096) uint8_t page[8192];
  const uint64_t base = offset & ~uint64_t(4095);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + 4095) / 4096) * 4096;
  ssize_t got;
  do { got = pread(s->fd, page, requested, base); } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) fail("short or failed direct read");
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
}

extern "C" Store *xtier_rows_open(const char *path, uint64_t rows,
                                 uint64_t woff, uint64_t soff, uint64_t budget) {
  auto *s = new Store;
  const char *mode = std::getenv("XTIER_ENGRAM_MODE");
  const bool ram = mode && std::strcmp(mode, "ram") == 0;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (ram ? 0 : O_DIRECT));
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat statbuf;
  if (fstat(s->fd, &statbuf) || woff > uint64_t(statbuf.st_size) ||
      soff > uint64_t(statbuf.st_size) || rows > (uint64_t(statbuf.st_size)-woff)/256 ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");
  if (ram) {
    s->resident_size = statbuf.st_size;
    s->resident = static_cast<uint8_t *>(mmap(nullptr,s->resident_size,
        PROT_READ,MAP_SHARED | MAP_POPULATE,s->fd,0));
    if (s->resident == MAP_FAILED) fail("resident mapping");
    if (mlock(s->resident,s->resident_size)) fail("RAM mode requires memlock capability and sufficient RAM");
    budget = 0;
  }
  s->rows = rows; s->weight_offset = woff; s->scale_offset = soff;
  s->row_lo = 0; s->row_hi = rows;
  s->slots = budget / (264 + sizeof(uint64_t));
  s->cache = nullptr; s->keys = nullptr;
  if (s->slots) {
    s->cache = static_cast<uint8_t *>(mmap(nullptr, s->slots * 264,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->keys = static_cast<uint64_t *>(mmap(nullptr, s->slots * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (s->cache == MAP_FAILED || s->keys == MAP_FAILED) fail("cache allocation");
  }
  if (!ram) {
    if (posix_memalign(reinterpret_cast<void **>(&s->pages), 4096, kBatchPages * 4096))
      fail("batch buffer allocation");
    if (syscall(SYS_io_setup, kBatchPages, &s->aio) < 0) fail("io_setup");
  }
  return s;
}

static void xtier_rows_lookup_serial(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row ID out of bounds");
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(work->weights + i * 256, 0, 256);
      std::memset(work->scales + i * 8, 0, 8);
      continue;
    }
    uint8_t row[264];
    const uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
    std::unique_lock<std::mutex> guard(s->locks[slot % 256]);
    if (s->slots && s->keys[slot] == uint64_t(id) + 1) {
      std::memcpy(row, s->cache + slot * 264, 264);
      ++s->hits;
    } else {
      read_bytes(s, s->weight_offset + uint64_t(id) * 256, row, 256);
      read_bytes(s, s->scale_offset + uint64_t(id) * 8, row + 256, 8);
      ++s->misses;
      if (s->slots) {
        std::memcpy(s->cache + slot * 264, row, 264);
        s->keys[slot] = uint64_t(id) + 1;
      }
    }
    guard.unlock();
    std::memcpy(work->weights + i * 256, row, 256);
    std::memcpy(work->scales + i * 8, row + 256, 8);
  }
}


// At most 32 misses and 128 unique pages per batch, including boundary crossings.
struct ReadBatch {
  std::array<uint64_t, 128> offsets{};
  std::array<size_t, 128> required{};
  size_t count = 0;
  void add(uint64_t offset, size_t length) {
    while (length) {
      uint64_t base = offset & ~uint64_t(4095);
      size_t delta = offset - base, take = std::min(length, 4096 - delta);
      size_t j = 0;
      while (j < count && offsets[j] != base) ++j;
      if (j == count) {
        if (count == offsets.size()) fail("batch page limit");
        offsets[count++] = base;
      }
      required[j] = std::max(required[j], delta + take);
      offset += take; length -= take;
    }
  }
  void read(Store *s) {
    if (!count) return;
    std::array<iocb, 128> requests{};
    std::array<iocb *, 128> pointers{};
    for (size_t j = 0; j < count; ++j) {
      auto &request = requests[j];
      request.aio_data = j;
      request.aio_lio_opcode = IOCB_CMD_PREAD;
      request.aio_fildes = s->fd;
      request.aio_buf = reinterpret_cast<uint64_t>(s->pages + j * 4096);
      request.aio_nbytes = 4096;
      request.aio_offset = offsets[j];
      pointers[j] = &request;
    }
    size_t submitted = 0;
    while (submitted < count) {
      long queued;
      do { queued = syscall(SYS_io_submit, s->aio, count - submitted,
                            pointers.data() + submitted); }
      while (queued < 0 && errno == EINTR);
      if (queued <= 0) fail("io_submit");
      std::array<io_event, 128> events{};
      long completed = 0;
      while (completed < queued) {
        long got;
        do { got = syscall(SYS_io_getevents, s->aio, 1, queued - completed,
                           events.data(), nullptr); }
        while (got < 0 && errno == EINTR);
        if (got <= 0) fail("io_getevents");
        for (long j = 0; j < got; ++j) {
          const auto &event = events[j];
          if (event.data < submitted || event.data >= submitted + queued ||
              event.res < 0 || event.res2 != 0 ||
              uint64_t(event.res) < required[event.data])
            fail("short or failed asynchronous read");
        }
        completed += got;
      }
      submitted += queued;
    }
    s->reads.fetch_add(count, std::memory_order_relaxed);
    ++s->batches;
    if (count > s->peak_pages.load()) s->peak_pages.store(count);
  }
  void copy(Store *s, uint64_t offset, uint8_t *out, size_t length) const {
    while (length) {
      uint64_t base = offset & ~uint64_t(4095);
      size_t delta = offset - base, take = std::min(length, 4096 - delta);
      size_t j = 0;
      while (j < count && offsets[j] != base) ++j;
      if (j == count) fail("missing batch page");
      std::memcpy(out, s->pages + j * 4096 + delta, take);
      offset += take; out += take; length -= take;
    }
  }
};

static uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

static void xtier_rows_lookup_batched(void *opaque);

// Time spent inside the callback (cache, I/O, copies); the GPU-visible gap minus this
// is the callback's dispatch latency.
extern "C" void xtier_rows_lookup(void *opaque) {
  const uint64_t start = now_ns();
  Store *s = static_cast<Work *>(opaque)->store;
  if (s->resident) xtier_rows_lookup_serial(opaque);
  else xtier_rows_lookup_batched(opaque);
  const uint64_t took = now_ns() - start;
  s->lookup_ns.fetch_add(took, std::memory_order_relaxed);
  s->lookups.fetch_add(1, std::memory_order_relaxed);
  uint64_t seen = s->lookup_max_ns.load(std::memory_order_relaxed);
  while (took > seen && !s->lookup_max_ns.compare_exchange_weak(seen, took)) {}
}

extern "C" void xtier_rows_timing(Store *s, uint64_t *out) {
  out[0] = s->lookup_ns.load(); out[1] = s->lookups.load(); out[2] = s->lookup_max_ns.load();
}

// Persistent reader threads (XTIER_ENGRAM_IO_THREADS, default 16): the O_DIRECT reads of one lookup
// run concurrently, so the NVMe sees real queue depth (native AIO here completed them one by one).
// The calling (CUDA callback) thread takes part as well. One job at a time.
struct IoPool {
  std::mutex job_lock, m;
  std::condition_variable start, finish;
  std::vector<std::thread> threads;
  uint64_t generation = 0;
  int fd = -1;
  const uint64_t *offsets = nullptr;
  const uint32_t *required = nullptr;
  uint8_t *pages = nullptr;
  size_t count = 0;
  std::atomic<size_t> next{0};
  size_t active = 0;
  std::atomic<bool> failed{false};

  void run_share() {
    for (size_t j; (j = next.fetch_add(1)) < count;) {
      ssize_t got;
      do { got = pread(fd, pages + j * 4096, 4096, offsets[j]); } while (got < 0 && errno == EINTR);
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
    const char *env = std::getenv("XTIER_ENGRAM_IO_THREADS");
    int n = env ? std::atoi(env) : 16;
    for (int k = 0; k < std::max(0, n); ++k) threads.emplace_back([this] { worker(); });
    for (auto &t : threads) t.detach();
  }
  void read(int fd_, const uint64_t *offsets_, const uint32_t *required_, uint8_t *pages_, size_t count_) {
    std::lock_guard<std::mutex> job(job_lock);
    {
      std::lock_guard<std::mutex> lock(m);
      fd = fd_; offsets = offsets_; required = required_; pages = pages_; count = count_;
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
  static IoPool *pool = new IoPool();   // never destroyed: threads are detached
  return *pool;
}

// Read `count` 4 KiB pages at `offsets` into s->pages[0..count); page j must return at least
// required[j] bytes.
static void read_pages(Store *s, const uint64_t *offsets, const uint32_t *required, size_t count) {
  if (!count) return;
  io_pool().read(s->fd, offsets, required, s->pages, count);
  s->reads.fetch_add(count, std::memory_order_relaxed);
  ++s->batches;
  if (count > s->peak_pages.load()) s->peak_pages.store(count);
}

// Copy `length` bytes starting at file offset `offset` out of consecutive pages from `page`.
static void copy_pages(const Store *s, uint32_t page, uint64_t offset, uint8_t *out, size_t length) {
  size_t delta = offset & 4095;
  while (length) {
    size_t take = std::min(length, size_t(4096) - delta);
    std::memcpy(out, s->pages + size_t(page) * 4096 + delta, take);
    out += take; length -= take; delta = 0; ++page;
  }
}

struct Miss { uint64_t index, id; uint32_t weight_page, scale_page; };

static void xtier_rows_lookup_batched(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  // One callback owns the I/O buffer and the cache until every read completes.
  std::lock_guard<std::mutex> guard(s->batch_lock);
  static thread_local std::array<uint64_t, kBatchPages> offsets;
  static thread_local std::array<uint32_t, kBatchPages> required;
  static thread_local std::array<Miss, kBatchPages> misses;
  size_t pages = 0, miss_count = 0;
  auto add_pages = [&](uint64_t offset, size_t length) {
    const uint32_t first = uint32_t(pages);
    uint64_t base = offset & ~uint64_t(4095);
    size_t need = (offset - base) + length;
    while (need) {
      offsets[pages] = base;
      required[pages] = uint32_t(std::min(need, size_t(4096)));
      ++pages; base += 4096; need -= std::min(need, size_t(4096));
    }
    return first;
  };
  auto flush = [&]() {
    read_pages(s, offsets.data(), required.data(), pages);
    for (size_t k = 0; k < miss_count; ++k) {
      const Miss &m = misses[k];
      uint8_t row[264];
      copy_pages(s, m.weight_page, s->weight_offset + m.id * 256, row, 256);
      if (s->scale_data) std::memcpy(row + 256, s->scale_data + (m.id - s->scale_row_lo) * 8, 8);
      else copy_pages(s, m.scale_page, s->scale_offset + m.id * 8, row + 256, 8);
      std::memcpy(work->weights + m.index * 256, row, 256);
      std::memcpy(work->scales + m.index * 8, row + 256, 8);
      if (s->slots) {
        uint64_t slot = m.id % s->slots;
        std::memcpy(s->cache + slot * 264, row, 264);
        s->keys[slot] = m.id + 1;
      }
    }
    pages = miss_count = 0;
  };
  for (uint64_t i = 0; i < work->count; ++i) {
    int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row ID out of bounds");
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(work->weights + i * 256, 0, 256);
      std::memset(work->scales + i * 8, 0, 8);
      continue;
    }
    uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
    if (s->slots && s->keys[slot] == uint64_t(id) + 1) {
      std::memcpy(work->weights + i * 256, s->cache + slot * 264, 256);
      std::memcpy(work->scales + i * 8, s->cache + slot * 264 + 256, 8);
      ++s->hits;
      continue;
    }
    // a row needs at most 2 weight pages and 2 scale pages
    if (pages + 4 > kBatchPages || miss_count == kBatchPages) flush();
    Miss m{i, uint64_t(id), 0, 0};
    m.weight_page = add_pages(s->weight_offset + uint64_t(id) * 256, 256);
    if (!s->scale_data) m.scale_page = add_pages(s->scale_offset + uint64_t(id) * 8, 8);
    misses[miss_count++] = m;
    ++s->misses;
  }
  flush();
}


// Explicit post-load operation: publish only after the exact scales of the owned rows
// [row_lo, row_hi) are read (the whole table on one GPU; a TP rank never looks up other rows).
extern "C" int xtier_rows_preload_scales(Store *s) {
  std::lock_guard<std::mutex> guard(s->batch_lock);
  if (s->scale_data || s->resident) return 0;
  const uint64_t first = s->scale_offset + s->row_lo * 8;
  const uint64_t base = first & ~uint64_t(4095);
  const size_t delta = first - base, total = delta + (s->row_hi - s->row_lo) * 8;
  const size_t allocation = (total + 4095) & ~size_t(4095);
  auto *data = static_cast<uint8_t *>(mmap(nullptr, allocation,
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (data == MAP_FAILED) return -1;
  for (size_t done = 0; done < total;) {
    size_t requested = std::min(size_t(8 * 1024 * 1024), allocation - done);
    ssize_t got;
    do { got = pread(s->fd, data + done, requested, base + done); }
    while (got < 0 && errno == EINTR);
    if (got < 0 || size_t(got) < std::min(requested, total - done)) {
      int error = got < 0 ? errno : EIO;
      munmap(data, allocation); errno = error; return -1;
    }
    done += requested;
  }
  s->scale_storage = data; s->scale_allocation = allocation;
  s->scale_row_lo = s->row_lo; s->scale_row_hi = s->row_hi;
  s->scale_data = data + delta; s->scale_bytes.store((s->row_hi - s->row_lo) * 8);
  return 0;
}

extern "C" void xtier_rows_io_stats(Store *s, uint64_t *out) {
  out[0] = s->batches.load(); out[1] = s->peak_pages.load();
  out[2] = s->scale_bytes.load();
}

extern "C" void xtier_rows_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load(); out[1] = s->misses.load(); out[2] = s->reads.load();
  out[3] = s->slots * 272;
}
extern "C" void xtier_rows_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  if (s->scale_data && (lo < s->scale_row_lo || hi > s->scale_row_hi))
    fail("row ownership range outside the preloaded scales");
  s->row_lo = lo; s->row_hi = hi;
}
extern "C" void xtier_rows_close(Store *s) {
  if (s->resident) munmap(s->resident,s->resident_size);
  if (s->slots) {
    munmap(s->cache, s->slots * 264);
    munmap(s->keys, s->slots * sizeof(uint64_t));
  }
  if (s->aio && syscall(SYS_io_destroy, s->aio) < 0) fail("io_destroy");
  if (s->scale_storage) munmap(s->scale_storage, s->scale_allocation);
  std::free(s->pages);
  close(s->fd); delete s;
}
