// The CPU engine (csrc/engine) and the cold store (csrc/cold) under a sanitizer, without a GPU (tests/engine_san.sh
// builds and runs it). A made-up NVFP4 layer (H 1024, I 512, 64 experts, top-6):
//   plan A  every expert in RAM;
//   plan B  half of them in RAM, the others in a store of two files (cold experts, read by the forwards).
// Three threads at once, as a server has them: one caller per plan (forwards of 1..256 tokens over random routes, the
// decode path's and the prefill's) and one that pins / unpins B's slots (the GPU pool's policy thread). B's outputs are
// compared with A's. Then the error paths: a route outside the table, calls that route to more experts than B has RAM
// slots (they fail, and every slot stays usable: 2026-11-10 audit R5), a store file cut short under the forwards.
//   engine_san THREADS DIR [ITERATIONS=300]
#include "xtier_engine.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static const int H = 1024, I = 512, E = 64, K = 6;
static std::atomic<int> bad{0};
#define CHECK(ok, ...) do { if (!(ok)) { bad++; std::fprintf(stderr, "FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

struct Call {
  int m;
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> rw;
};

static Call make(std::mt19937 &g, int m, int experts = E) {
  Call c{m, std::vector<uint16_t>(size_t(m) * H), std::vector<int32_t>(size_t(m) * K), std::vector<float>(size_t(m) * K)};
  std::uniform_int_distribution<int> small(0, 255);
  for (auto &v : c.x) v = uint16_t(0x3C00 | small(g));   // BF16 values near 1 / -1 would overflow nothing
  for (auto &v : c.x) if (small(g) & 1) v |= 0x8000;
  for (int t = 0; t < m; t++) {
    float sum = 0;
    for (int q = 0; q < K; q++) {
      int e;
      bool dup;
      do {
        e = int(g() % unsigned(experts));
        dup = false;
        for (int p = 0; p < q; p++) dup |= c.ids[size_t(t) * K + p] == e;
      } while (dup);
      c.ids[size_t(t) * K + q] = e;
      c.rw[size_t(t) * K + q] = float(small(g) + 1);
      sum += c.rw[size_t(t) * K + q];
    }
    for (int q = 0; q < K; q++) c.rw[size_t(t) * K + q] /= sum;
  }
  return c;
}

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: engine_san THREADS DIR [ITERATIONS]\n"); return 2; }
  const int threads = std::atoi(argv[1]), iterations = argc > 3 ? std::atoi(argv[3]) : 300;
  const std::string dir = argv[2];
  std::mt19937 g(20261110);
  std::vector<uint8_t> w13(size_t(E) * 2 * I * (H / 2)), w2(size_t(E) * H * (I / 2)), s13(size_t(E) * 2 * I * (H / 16)),
      s2(size_t(E) * H * (I / 16));
  for (auto &v : w13) v = uint8_t(g());
  for (auto &v : w2) v = uint8_t(g());
  for (auto &v : s13) v = uint8_t(0x28 + g() % 16);   // E4M3, positive and small
  for (auto &v : s2) v = uint8_t(0x28 + g() % 16);
  std::vector<float> g13(size_t(E) * 2), g2(E);
  for (auto &v : g13) v = 0.5f + float(g() % 100) / 100.f;
  for (auto &v : g2) v = 0.5f + float(g() % 100) / 100.f;

  void *A = xtier_create3(H, I, E, threads, 10.f, 0, w13.data(), w2.data(), s13.data(), s2.data(), g13.data(), g2.data());
  if (!A) { std::fprintf(stderr, "create A: %s\n", xtier_last_error()); return 1; }
  const int nodes = xtier_nodes(A);
  std::printf("isa %s, %d worker nodes, %d threads\n", xtier_isa(), nodes, threads);

  // plan B: the first half of the experts in RAM, a record of every expert in the store
  const int slots = 48;   // 16 experts on the drives; a decode call of up to 7 tokens (42 experts) always fits
  std::vector<int32_t> resident(slots);
  for (int j = 0; j < slots; j++) resident[j] = j;
  void *B = xtier_create4(H, I, E, slots, threads, 10.f, 0, w13.data(), w2.data(), s13.data(), s2.data(), g13.data(),
                          g2.data(), resident.data());
  if (!B) { std::fprintf(stderr, "create B: %s\n", xtier_last_error()); return 1; }
  const size_t IN = I / nodes, HN = H / nodes;
  const size_t sizes[6] = {IN * (H / 2), IN * (H / 2), IN * (H / 16), IN * (H / 16), HN * (I / 2), HN * (I / 16)};
  size_t part = 0;
  for (size_t s : sizes) part += s;
  const std::string paths[2] = {dir + "/san-0.cold", dir + "/san-1.cold"};
  const char *names[2] = {paths[0].c_str(), paths[1].c_str()};
  void *store = xtier_cold_open(names, 2, 4, 1);
  if (!store) { std::fprintf(stderr, "store: %s\n", xtier_cold_error()); return 1; }
  std::vector<int32_t> files(size_t(E) * nodes);
  std::vector<long long> offsets(size_t(E) * nodes);
  for (int e = 0; e < E; e++)
    for (int n = 0; n < nodes; n++) {
      const uint8_t *w13e = w13.data() + size_t(e) * 2 * I * (H / 2), *s13e = s13.data() + size_t(e) * 2 * I * (H / 16);
      const uint8_t *w2e = w2.data() + size_t(e) * H * (I / 2), *s2e = s2.data() + size_t(e) * H * (I / 16);
      const void *pieces[6] = {w13e + n * IN * (H / 2),  w13e + (I + n * IN) * (H / 2), s13e + n * IN * (H / 16),
                               s13e + (I + n * IN) * (H / 16), w2e + n * HN * (I / 2), s2e + n * HN * (I / 16)};
      const int f = n % 2;
      const long long offset = (long long)(size_t(e) * (nodes / 2) + n / 2) * (long long)part;
      files[size_t(e) * nodes + n] = f;
      offsets[size_t(e) * nodes + n] = offset;
      if (xtier_cold_write(store, f, offset, pieces, sizes, 6)) { std::fprintf(stderr, "write: %s\n", xtier_cold_error()); return 1; }
    }
  if (xtier_cold_attach(B, store, files.data(), offsets.data(), nullptr, 50.f)) {
    std::fprintf(stderr, "attach: %s\n", xtier_last_error());
    return 1;
  }

  // the calls, and A's answers to them before the threads start
  const int shapes[] = {1, 2, 4, 6, 8, 8, 6, 37, 256};
  std::vector<Call> calls;
  for (int i = 0; i < iterations; i++) calls.push_back(make(g, shapes[i % 9]));
  std::vector<std::vector<float>> want(calls.size());
  for (size_t i = 0; i < calls.size(); i++) {
    auto &c = calls[i];
    want[i].resize(size_t(c.m) * H);
    const int rc = xtier_forward(A, c.m, K, c.x.data(), c.ids.data(), c.rw.data(), want[i].data());
    CHECK(rc >= 0, "A forward %zu: %s", i, xtier_last_error());
  }
  std::atomic<bool> stop{false};
  std::atomic<long> pins{0};
  std::thread pinner([&] {   // the GPU pool's policy thread: keeps experts in RAM while it copies them
    std::mt19937 h(7);
    while (!stop.load()) {
      const int slot = xtier_slot_pin(B, int(h() % E));
      if (slot >= 0) { pins++; std::this_thread::yield(); xtier_slot_unpin(B, slot); }
    }
  });
  std::thread a([&] {   // plan A again, from its own caller: the same bits every time
    std::vector<float> y;
    for (size_t i = 0; i < calls.size(); i++) {
      auto &c = calls[i];
      y.assign(size_t(c.m) * H, 0.f);
      const int rc = c.m <= 8 ? xtier_forward_decode(A, c.m, K, c.x.data(), c.ids.data(), c.rw.data(), y.data())
                              : xtier_forward(A, c.m, K, c.x.data(), c.ids.data(), c.rw.data(), y.data());
      CHECK(rc >= 0, "A forward %zu again: %s", i, xtier_last_error());
      CHECK(!std::memcmp(y.data(), want[i].data(), y.size() * 4), "A call %zu: not the same bits twice", i);
    }
  });
  double worst = 0;
  long cold_calls = 0;
  {   // plan B from this thread: the cold experts are read as the forwards need them
    std::vector<float> y;
    for (size_t i = 0; i < calls.size(); i++) {
      auto &c = calls[i];
      y.assign(size_t(c.m) * H, 0.f);
      int rc = -1;
      if (c.m * K > slots - 2) continue;   // more experts than RAM slots: Python splits such calls (CpuPlan.forward)
      rc = c.m <= 8 ? xtier_forward_decode(B, c.m, K, c.x.data(), c.ids.data(), c.rw.data(), y.data())
                    : xtier_forward(B, c.m, K, c.x.data(), c.ids.data(), c.rw.data(), y.data());
      CHECK(rc >= 0, "B forward %zu (m %d): %s", i, c.m, xtier_last_error());
      if (rc < 0) continue;
      cold_calls++;
      double d2 = 0, r2 = 0;
      for (size_t j = 0; j < y.size(); j++) {
        const double d = double(y[j]) - want[i][j];
        d2 += d * d; r2 += double(want[i][j]) * want[i][j];
        if (!std::isfinite(y[j])) { d2 = INFINITY; break; }
      }
      const double rel = std::sqrt(d2 / std::max(r2, 1e-30));
      worst = std::max(worst, rel);
      CHECK(rel < 1e-4, "B call %zu (m %d): rel L2 %.3e against A", i, c.m, rel);
    }
  }
  a.join();
  double stats[7];
  xtier_cold_plan_stats(B, stats);
  std::printf("B: %ld calls, %.0f experts read from the store, worst rel L2 against A %.2e; %ld pins meanwhile\n", cold_calls,
              stats[1], worst, pins.load());

  // error paths. 1: a route outside the table
  {
    Call c = make(g, 4);
    c.ids[5] = E + 3;
    std::vector<float> y(size_t(4) * H);
    const int rc = xtier_forward(B, 4, K, c.x.data(), c.ids.data(), c.rw.data(), y.data());
    CHECK(rc < 0, "a route outside the table was accepted");
    std::printf("route outside the table: rc %d (%s)\n", rc, rc < 0 ? xtier_last_error() : "");
    Call ok = make(g, 4);
    CHECK(xtier_forward(B, 4, K, ok.x.data(), ok.ids.data(), ok.rw.data(), y.data()) >= 0, "B after the bad route: %s", xtier_last_error());
  }
  // 1b: a call that routes to more experts than there are RAM slots (24 tokens x 6 routes over 64 experts: about 57 of them, 48 slots):
  //     it must fail and leave every slot usable
  {
    std::vector<int32_t> table(E);
    auto in_ram = [&] { xtier_slot_table(B, table.data()); int n = 0; for (int e = 0; e < E; e++) n += table[e] >= 0; return n; };
    const int before = in_ram();
    int failed = 0;
    std::vector<float> y(size_t(24) * H);
    for (int i = 0; i < 12; i++) {
      Call c = make(g, 24);
      failed += xtier_forward(B, 24, K, c.x.data(), c.ids.data(), c.rw.data(), y.data()) < 0;
    }
    const int after = in_ram();
    std::printf("calls with more experts than RAM slots: %d of 12 failed (%s); experts in RAM %d -> %d of %d slots\n", failed,
                failed ? xtier_last_error() : "", before, after, slots);
    int later = 0;
    for (int i = 0; i < 60; i++) {
      Call c = make(g, 4);
      later += xtier_forward_decode(B, 4, K, c.x.data(), c.ids.data(), c.rw.data(), y.data()) < 0;
    }
    std::printf("  ordinary 4-token calls after that: %d of 60 failed%s%s\n", later, later ? ": " : "", later ? xtier_last_error() : "");
    xtier_slot_table(B, table.data());
    CHECK(later == 0, "RAM slots lost after calls that did not fit: %d of 60 ordinary calls fail", later);
  }
  // 2: a store file cut short (a drive that drops out): forwards that need its experts must fail, never compute
  //    with what was in the slot; the forwards after it must work again once the file is back
  {
    std::vector<uint8_t> keep;
    const int fd = open(paths[1].c_str(), O_RDWR);
    const off_t size = lseek(fd, 0, SEEK_END);
    keep.resize(size_t(size));
    CHECK(pread(fd, keep.data(), keep.size(), 0) == ssize_t(size), "reading the store file back");
    CHECK(ftruncate(fd, 4096) == 0, "truncate");
    int failed = 0, fine = 0;
    std::vector<float> y(size_t(8) * H);
    for (int i = 0; i < 40; i++) {
      Call c = make(g, 4);
      const int rc = xtier_forward_decode(B, 4, K, c.x.data(), c.ids.data(), c.rw.data(), y.data());
      if (rc < 0) failed++; else fine++;
    }
    std::printf("store file cut short: %d forwards failed (%s), %d needed no read\n", failed, xtier_last_error(), fine);
    CHECK(failed > 0, "no forward failed with the store file cut short");
    CHECK(pwrite(fd, keep.data(), keep.size(), 0) == ssize_t(size), "writing the store file back");
    fsync(fd);
    close(fd);
    std::vector<int32_t> table(E);
    const int have = xtier_slot_table(B, table.data());
    int in_ram = 0;
    for (int e = 0; e < E; e++) in_ram += table[e] >= 0;
    std::printf("after the failures: %d slots, %d experts in RAM\n", have, in_ram);
    int again = 0;
    double worst2 = 0;
    for (int i = 0; i < 60; i++) {
      Call c = make(g, 4);
      std::vector<float> ref(size_t(4) * H);
      CHECK(xtier_forward(A, 4, K, c.x.data(), c.ids.data(), c.rw.data(), ref.data()) >= 0, "A reference");
      const int rc = xtier_forward_decode(B, 4, K, c.x.data(), c.ids.data(), c.rw.data(), y.data());
      if (rc < 0) { again++; continue; }
      double d2 = 0, r2 = 0;
      for (size_t j = 0; j < size_t(4) * H; j++) { const double d = double(y[j]) - ref[j]; d2 += d * d; r2 += double(ref[j]) * ref[j]; }
      worst2 = std::max(worst2, std::sqrt(d2 / std::max(r2, 1e-30)));
    }
    std::printf("with the file back: %d of 60 forwards failed, worst rel L2 against A %.2e\n", again, worst2);
    CHECK(again == 0, "forwards still fail after the store file is back: %s", xtier_last_error());
    CHECK(worst2 < 1e-4, "wrong results after the failed reads: rel L2 %.3e", worst2);
  }
  stop = true;
  pinner.join();
  xtier_destroy(B);
  xtier_destroy(A);
  xtier_cold_close(store);
  unlink(paths[0].c_str());
  unlink(paths[1].c_str());
  std::printf("ENGINE-SAN %s\n", bad.load() ? "FAIL" : "PASS");
  return bad.load() ? 1 : 0;
}
