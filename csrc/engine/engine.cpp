// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments
// renamed only, so the kernels compile to the same code.
// Independent NVFP4 CPU MoE, generic over the model: hidden size H and expert
// intermediate size I (both multiples of 64), 10..512 experts, top-k<=10, optional
// SwiGLU clamp limit, set per plan at creation. Qwen3.8 (2560/640), DeepSeek V4.1
// (5120/2304) and GLM-5.3 (4096/2048) shapes are template-specialised from the same code;
// other shapes use the runtime instantiation. Decode uses <=8 tokens; synchronous prefill groups up to 2048.
// BF16 input; FP32 SiLU, intermediate, down projection, accumulation and output.
// Kernels: AVX512 BF16 (+VBMI) when compiled for it, else AVX2/FMA (Zen2 and other hosts
// without AVX512). CMake builds this file once per ISA (XTIER_IMPL_PREFIX set, one library each)
// and src/dispatch.cpp loads one at run time; a plain single build keeps the xtier_* names.
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <linux/futex.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX512BF16__) && defined(__AVX512BW__)
#define XTIER_AVX512 1
#elif defined(__AVX2__) && defined(__FMA__)
#define XTIER_AVX512 0
#else
#error "xtier needs AVX512 BF16/BW or AVX2+FMA (e.g. -march=x86-64-v3)"
#endif

namespace {
#ifndef XTIER_PREFILL_MICRO_TOKENS
#define XTIER_PREFILL_MICRO_TOKENS 8  // tokens per expert group (4 or 8)
#endif
#ifndef XTIER_DOWN_F32_LUT
#define XTIER_DOWN_F32_LUT 1  // Down: unpack NVFP4 straight to FP32 (same values and order)
#endif
#ifndef XTIER_VBMI_UNPACK
#define XTIER_VBMI_UNPACK 1          // VBMI FP4->BF16 unpack (bitwise identical to the generic one)
#endif
#ifndef XTIER_LANE_TIMING
#define XTIER_LANE_TIMING 0          // offline: print per-phase worker idle at the barriers (bench/ds41_lane_timing.py)
#endif
#ifndef XTIER_WIDE_ROWS_M
#define XTIER_WIDE_ROWS_M 6  // largest M>4 that still processes two rows per iteration
#endif
// Each switch below preserves every FP32 operation and accumulation order.
#ifndef XTIER_BALANCED_SMALL_DOWN
#define XTIER_BALANCED_SMALL_DOWN 1  // M=2..4: equal contiguous Down rows per worker
#endif
#ifndef XTIER_BALANCED_GATE
#define XTIER_BALANCED_GATE 1        // M<=4: cost-balanced contiguous Gate rows per worker
#endif
#ifndef XTIER_GATE_COST_A
#define XTIER_GATE_COST_A 2          // per-row cost = A + tokens sharing the row
#endif
#ifndef XTIER_HUGEPAGES
#define XTIER_HUGEPAGES 1            // MADV_HUGEPAGE on NUMA-bound weight mappings
#endif
#ifndef XTIER_NODE_BARRIER
#define XTIER_NODE_BARRIER 1         // node leaders gather local workers first
#endif
// Down rows per microkernel iteration for M<=4 (4, 2 or 1). Four rows of I/2 bytes interleave four
// short weight streams; 1 reads each row through in one stream, which DeepSeek V4.1 1017 MXFP4
// (I=2304) needs: Down -15% with XTIER_DOWN_GRAIN=24 (research/mxfp4/DOWN_R4.md). Row results do
// not depend on it.
#ifndef XTIER_DOWN_R4
#define XTIER_DOWN_R4 4
#endif
static_assert(XTIER_DOWN_R4==1||XTIER_DOWN_R4==2||XTIER_DOWN_R4==4,"XTIER_DOWN_R4 must be 1, 2 or 4");
constexpr int MAX_M=XTIER_PREFILL_MICRO_TOKENS,MAX_K=10,MAX_BATCH=2048,MAX_EXPERTS=512,
              MAX_ROUTES=MAX_EXPERTS+(MAX_BATCH*MAX_K+MAX_M-1)/MAX_M,MAX_NODES=16,DECODE_M=8;
static_assert(MAX_M==4||MAX_M==8,"XTIER_PREFILL_MICRO_TOKENS must be 4 or 8");
constexpr float fp4[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
constexpr uint16_t tobf(float v) { uint32_t u=__builtin_bit_cast(uint32_t,v);u+=0x7fff+((u>>16)&1);return uint16_t(u>>16); }
constexpr float pow2(int k) { float r=1;for(;k>0;k--)r*=2;for(;k<0;k++)r*=.5f;return r; }
// E4M3 value; same as ldexp(1+m/8,e-7) / ldexp(m,-9), exact.
constexpr float scale(int s) { int e=s>>3,m=s&7;return e ? (1.f+m/8.f)*pow2(e-7) : float(m)*pow2(-9); }
// Tables are constant-initialised: no load-time code runs from a build for an ISA the
// host may lack (both ISA builds live in one library).
alignas(64) constexpr auto lut=[] {
    std::array<std::array<uint16_t,16>,128> t{};
    for(int s=0;s<127;s++)for(int q=0;q<16;q++)t[s][q]=tobf(scale(s)*fp4[q]);
    return t;
}();

#if XTIER_AVX512
#if defined(__AVX512VBMI__) && XTIER_VBMI_UNPACK
// Same BF16 lanes as the generic unpack below, bit for bit, with three shuffles instead of six
// (Zen4 shuffle ports bound the old one: ~11.5 vs ~16 GB/s per core on a Down row loop).
// Output byte b is byte b%2 of weight j=b/2. After the vpermq, qword b/8 holds source bytes
// 0..7 (b<32) or 8..15, so nibble j sits at bit 8*((j/2)%8)+4*(j%2) of its qword; multishift
// rotates it to bits 1..4, and the index (bits&0x1E)|(b%2)|64*(j>=16) picks byte 2*code+b%2
// of the scale-s0 table or (bit 6) the scale-s1 table in vpermt2b.
alignas(64) constexpr auto unpack_shift=[] {
    std::array<uint8_t,64> t{};
    for(int b=0;b<64;b++){int j=b/2;t[b]=uint8_t((8*((j/2)%8)+4*(j%2)+63)&63);}
    return t;
}();
alignas(64) constexpr auto unpack_select=[] {
    std::array<uint8_t,64> t{};
    for(int b=0;b<64;b++)t[b]=uint8_t((b%2)|(b/2>=16?64:0));
    return t;
}();
inline __m512bh unpack(const uint8_t *p,const uint8_t *s, const std::array<uint16_t,16>* table=lut.data(), bool mx=false) {
    __m512i v=_mm512_permutexvar_epi64(_mm512_setr_epi64(0,0,0,0,1,1,1,1),_mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)p)));
    __m512i idx=_mm512_ternarylogic_epi32(_mm512_multishift_epi64_epi8(_mm512_load_si512(unpack_shift.data()),v),
                                          _mm512_set1_epi8(0x1E),_mm512_load_si512(unpack_select.data()),0xEA);  // (a&b)|c
    __m512i t0=_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)table[s[0]].data()));
    __m512i t1=_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)table[s[mx?0:1]].data()));
    return (__m512bh)_mm512_permutex2var_epi8(t0,idx,t1);
}
#else
inline __m512bh unpack(const uint8_t *p,const uint8_t *s, const std::array<uint16_t,16>* table=lut.data(), bool mx=false) {
    __m128i v=_mm_loadu_si128((const __m128i*)p),mask=_mm_set1_epi8(15);
    __m128i lo=_mm_and_si128(v,mask),hi=_mm_and_si128(_mm_srli_epi16(v,4),mask);
    __m256i codes=_mm256_set_m128i(_mm_unpackhi_epi8(lo,hi),_mm_unpacklo_epi8(lo,hi));
    __m512i idx=_mm512_cvtepu8_epi16(codes);
    idx=_mm512_mask_add_epi16(idx,0xffff0000u,idx,_mm512_set1_epi16(16));
    __m512i values=_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)table[s[0]].data()));
    values=_mm512_inserti64x4(values,_mm256_loadu_si256((const __m256i*)table[s[mx?0:1]].data()),1);
    return (__m512bh)_mm512_permutexvar_epi16(idx,values);
}
#endif
#endif  // XTIER_AVX512
// FP32 table per block scale, bit-for-bit the BF16 table widened: Down unpacks straight to
// FP32 lanes in the same weight order as asfloat(unpack()), with fewer instructions.
alignas(64) constexpr auto flut=[] {
    std::array<std::array<float,16>,128> t{};
    for(int s=0;s<128;s++)for(int q=0;q<16;q++)t[s][q]=__builtin_bit_cast(float,uint32_t(lut[s][q])<<16);
    return t;
}();
#if XTIER_AVX512
inline void unpack_f32(const uint8_t *p,const uint8_t *s,__m512 &lo,__m512 &hi, const std::array<float,16>* table=flut.data(), bool mx=false) {
    // Byte i holds weights 2i (low nibble) and 2i+1 (high nibble). Duplicate each byte into
    // two lanes, shift odd lanes by 4; vpermps reads only the low 4 index bits.
    const __m512i shift=_mm512_set_epi32(4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0);
    __m128i v=_mm_loadu_si128((const __m128i*)p);
    __m512i a=_mm512_srlv_epi32(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(v,v)),shift);
    __m512i b=_mm512_srlv_epi32(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(v,v)),shift);
    lo=_mm512_permutexvar_ps(a,_mm512_load_ps(table[s[0]].data()));
    hi=_mm512_permutexvar_ps(b,_mm512_load_ps(table[s[mx?0:1]].data()));
}
inline void asfloat(__m512bh a,__m512 &lo,__m512 &hi) {
    __m512i z=(__m512i)a;
    lo=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(z)),16));
    hi=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(z,1)),16));
}
#else
// AVX2: 32 weights (16 bytes, two E4M3 scales s[0], s[1]) -> FP32 weights 0..7, 8..15, 16..23,
// 24..31 in w[0..3], the BF16 table values widened. Per scale, two vpshufb pick the low and high
// bytes of the 16-entry BF16 table; the nibble split is shared by both blocks. Zen2 runs every
// byte shuffle on two FP pipes: vpermps or vcvtph2ps variants are slower there (llvm-mca znver2,
// bench/avx2_unpack_bench.cpp), and 32-weight steps halve the loop overhead.
// Low and high bytes of the 16 BF16 entries of each scale.
alignas(64) constexpr auto blut=[] {
    std::array<std::array<uint8_t,32>,128> t{};
    for(int s=0;s<128;s++)for(int q=0;q<16;q++){t[s][q]=uint8_t(lut[s][q]);t[s][16+q]=uint8_t(lut[s][q]>>8);}
    return t;
}();
inline void unpack32(const uint8_t *p,const uint8_t *s,__m256 w[4], const std::array<uint8_t,32>* table=blut.data(), bool mx=false) {
    __m128i v=_mm_loadu_si128((const __m128i*)p),mask=_mm_set1_epi8(15);
    __m128i lo=_mm_and_si128(v,mask),hi=_mm_and_si128(_mm_srli_epi16(v,4),mask);
    const __m128i codes[2]={_mm_unpacklo_epi8(lo,hi),_mm_unpackhi_epi8(lo,hi)};
    for(int b=0;b<2;b++) {
        const uint8_t *t=table[s[mx?0:b]].data();
        __m128i bl=_mm_shuffle_epi8(_mm_load_si128((const __m128i*)t),codes[b]);
        __m128i bh=_mm_shuffle_epi8(_mm_load_si128((const __m128i*)(t+16)),codes[b]);
        w[2*b]=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_unpacklo_epi8(bl,bh)),16));
        w[2*b+1]=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_unpackhi_epi8(bl,bh)),16));
    }
}
// 16 BF16 values -> FP32 (exact).
inline void bf16x16(const uint16_t *p,__m256 &lo,__m256 &hi) {
    __m256i v=_mm256_loadu_si256((const __m256i*)p);
    lo=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)),16));
    hi=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v,1)),16));
}
inline float hsum(__m256 v) {
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    s=_mm_add_ps(s,_mm_movehl_ps(s,s));
    return _mm_cvtss_f32(_mm_add_ss(s,_mm_movehdup_ps(s)));
}
// BF16 rounding kept in FP32 (XTIER_DOWN_BF16 intermediate on AVX2).
inline float round_bf16(float v) {return __builtin_bit_cast(float,uint32_t(tobf(v))<<16);}
#endif  // XTIER_AVX512

// Worker nodes. Every expert's Gate/Up rows (I) and Down rows (H) are split evenly over the
// worker nodes; node v's shard lives on NUMA node memory[v] and is computed by v's workers.
// Read from sysfs once: each NUMA node with CPUs is a worker node (NPS1/2/4, one or two
// sockets), cores = the first SMT thread of each of its physical cores, siblings = their
// second thread in the same order. XTIER_SPLIT_NODES=k (testing) splits every node's cores
// into k worker nodes sharing that node's memory. Without sysfs: the original fixed host
// (4 nodes, cores 16v..16v+15, siblings 64+16v..). The split only decides which worker
// computes a row; every row and every output element is computed with the same operations
// in the same order, so outputs are bitwise identical for any node count.
struct Topology {
    int nodes=0;
    int memory[MAX_NODES]{};
    std::vector<int> cores[MAX_NODES],siblings[MAX_NODES];
    std::string describe;
};
std::vector<int> parse_cpulist(const std::string &s) {
    std::vector<int> out;size_t i=0;
    while(i<s.size()) {
        if(!std::isdigit(static_cast<unsigned char>(s[i]))){i++;continue;}
        size_t j=i;while(j<s.size()&&std::isdigit(static_cast<unsigned char>(s[j])))j++;
        int lo=std::stoi(s.substr(i,j-i)),hi=lo;
        if(j<s.size()&&s[j]=='-'){size_t k=j+1;while(k<s.size()&&std::isdigit(static_cast<unsigned char>(s[k])))k++;hi=std::stoi(s.substr(j+1,k-j-1));j=k;}
        for(int c=lo;c<=hi;c++)out.push_back(c);
        i=j;
    }
    return out;
}
bool read_text(const std::string &path,std::string &out) {
    FILE *f=std::fopen(path.c_str(),"r");if(!f)return false;
    char buf[4096];size_t n=std::fread(buf,1,sizeof buf-1,f);std::fclose(f);buf[n]=0;out=buf;return true;
}
Topology detect_topology() {
    Topology t;
    std::string online;
    // XTIER_NODES=<NUMA node list> (e.g. 4-7): only these nodes become worker nodes. One process
    // per tensor-parallel rank uses its own nodes (python/xtier/tp.py sets it per rank).
    std::vector<int> only;
    if(const char *s=std::getenv("XTIER_NODES");s&&*s) {
        only=parse_cpulist(s);
        if(only.empty())throw std::runtime_error("Invalid XTIER_NODES");
    }
    if(read_text("/sys/devices/system/node/online",online)) {
        std::vector<int> nodes=parse_cpulist(online);
        for(int node:only)
            if(std::find(nodes.begin(),nodes.end(),node)==nodes.end())
                throw std::runtime_error("XTIER_NODES: NUMA node "+std::to_string(node)+" is not online");
        for(int node:nodes) {
            if(!only.empty()&&std::find(only.begin(),only.end(),node)==only.end())continue;
            std::string list;
            if(!read_text("/sys/devices/system/node/node"+std::to_string(node)+"/cpulist",list))continue;
            std::vector<int> cpus=parse_cpulist(list);
            if(cpus.empty())continue;   // memory-only node (e.g. CXL): no workers
            if(t.nodes==MAX_NODES)throw std::runtime_error("Too many NUMA nodes");
            int v=t.nodes++;t.memory[v]=node;
            for(int c:cpus) {
                std::string sib;std::vector<int> group{c};
                if(read_text("/sys/devices/system/cpu/cpu"+std::to_string(c)+"/topology/thread_siblings_list",sib))group=parse_cpulist(sib);
                if(group.empty()||*std::min_element(group.begin(),group.end())!=c)continue;   // not the first thread of its core
                t.cores[v].push_back(c);
                for(int g:group)if(g!=c){t.siblings[v].push_back(g);break;}
            }
            if(!t.siblings[v].empty()&&t.siblings[v].size()!=t.cores[v].size())t.siblings[v].clear();
        }
    }
    if(!t.nodes&&!only.empty())throw std::runtime_error("XTIER_NODES: none of the listed NUMA nodes has CPUs");
    if(!t.nodes) {
        t=Topology{};t.nodes=4;
        for(int v=0;v<4;v++){t.memory[v]=v;for(int j=0;j<16;j++){t.cores[v].push_back(v*16+j);t.siblings[v].push_back(64+v*16+j);}}
    }
    int split=1;
    if(const char *s=std::getenv("XTIER_SPLIT_NODES"))split=std::stoi(s);
    if(split<1||t.nodes*split>MAX_NODES)throw std::runtime_error("Invalid XTIER_SPLIT_NODES");
    if(split>1) {
        Topology u;
        for(int v=0;v<t.nodes;v++) {
            int n=int(t.cores[v].size());
            if(n%split)throw std::runtime_error("XTIER_SPLIT_NODES must divide every node's core count");
            for(int k=0;k<split;k++) {
                int w=u.nodes++;u.memory[w]=t.memory[v];
                for(int j=k*n/split;j<(k+1)*n/split;j++) {
                    u.cores[w].push_back(t.cores[v][j]);
                    if(!t.siblings[v].empty())u.siblings[w].push_back(t.siblings[v][j]);
                }
            }
        }
        t=std::move(u);
    }
    for(int v=0;v<t.nodes;v++) {
        t.describe+=(v?" | ":"")+std::string("node ")+std::to_string(t.memory[v])+": "+std::to_string(t.cores[v].size())+" cores "+
                    std::to_string(t.cores[v].front())+".."+std::to_string(t.cores[v].back());
        if(!t.siblings[v].empty())t.describe+=" + SMT "+std::to_string(t.siblings[v].front())+".."+std::to_string(t.siblings[v].back());
    }
    return t;
}
const Topology &topology() {static const Topology t=detect_topology();return t;}
// XTIER_NODE_CORES=C (default: all physical cores of a node): lanes 0..C-1 of a node use its
// first C physical cores, the remaining lanes their SMT siblings, so the node's other cores
// stay free for the host process (e.g. C=15 with 112 threads on 4 nodes: 15 cores + 13 siblings).
int node_cores(int threads) {
    const auto &t=topology();
    if(threads<t.nodes||threads>128||threads%t.nodes)
        throw std::runtime_error("Thread count must be a multiple of the "+std::to_string(t.nodes)+" worker nodes, at most 128");
    size_t phys=t.cores[0].size(),sibs=t.siblings[0].size();
    for(int v=1;v<t.nodes;v++){phys=std::min(phys,t.cores[v].size());sibs=std::min(sibs,t.siblings[v].size());}
    int cores=int(phys);
    if(const char *s=std::getenv("XTIER_NODE_CORES"))cores=std::stoi(s);
    int lanes=threads/t.nodes;
    if(cores<1||size_t(cores)>phys||size_t(lanes)>size_t(cores)+std::min(sibs,phys))
        throw std::runtime_error("XTIER_NODE_CORES="+std::to_string(cores)+" does not fit "+std::to_string(lanes)+
                                 " threads per node ("+std::to_string(phys)+" cores per node)");
    return cores;
}
// CPU of worker i (node i % nodes, lane i / nodes).
int worker_cpu(int i,int cores) {
    const auto &t=topology();
    int node=i%t.nodes,lane=i/t.nodes;
    return lane<cores?t.cores[node][lane]:t.siblings[node][lane-cores];
}

struct Free {size_t size=0;void operator()(uint8_t *p) const {if(p)munmap(p,size);}};
using Bytes=std::unique_ptr<uint8_t,Free>;
Bytes alloc(size_t n,int node) {
    size_t page=size_t(sysconf(_SC_PAGESIZE)),bytes=(n+page-1)/page*page;
    void *p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)throw std::bad_alloc();
    Bytes memory(static_cast<uint8_t*>(p),Free{bytes});
    unsigned long mask=1ul<<node;
    // Override an inherited process-wide interleave policy for these weights only.
    // Preferred, not bound: under memory pressure on this node (e.g. page cache from model
    // loading) pages spill to another node instead of invoking the OOM killer.
    if(syscall(SYS_mbind,p,bytes,MPOL_PREFERRED,&mask,sizeof(mask)*8,0))
        throw std::runtime_error("Binding weight shard to its NUMA node failed");
#if XTIER_HUGEPAGES
    // Advisory only: THP "madvise" policy; failure keeps 4KiB pages and identical math.
    // XTIER_THP=0 skips it: with defrag=madvise every huge-page fault may compact memory
    // synchronously, which stalls loading when RAM is nearly full (DeepSeek V4.1: ~12 s
    // per layer, while only ~12% of the shards ended up on huge pages anyway).
    static const bool thp=[]{const char *v=std::getenv("XTIER_THP");return !v||std::string(v)!="0";}();
    if(thp)madvise(p,bytes,MADV_HUGEPAGE);
#endif
    return memory; // Pages are committed by their owning workers during initialization.
}
struct Shard {Bytes gate,up,down,sg,su,sd;};
// row: first intermediate row of this group (prefix sum of counts; compact layout); slot: the RAM slot of the expert's weights.
struct Group {int expert=0,slot=0,count=0,row=0;int token[MAX_M]{};float route[MAX_M]{};};
struct alignas(64) Done {std::atomic<uint64_t> epoch{0};};

class Executor {
    int threads,nodes;
    int spin_limit=32768;
    // Spins before sleeping after the current call: the full limit after decode-sized calls
    // (the next layer's call follows within ~1 ms), a short one after prefill calls so idle
    // workers free their CPUs for the framework's host work (kernel launches, scheduling).
    std::atomic<int> spin_budget{32768};
    int prefill_spin=2048;
    std::vector<std::thread> workers;
    std::unique_ptr<Done[]> done;
    Done node_done[MAX_NODES];
    alignas(64) std::atomic<uint32_t> epoch{0};
    std::atomic<bool> quit{false};
    std::atomic<int> affinity_error{0};
    using Task=void(*)(void*,int,int,int);
    Task current=nullptr;
    void *context=nullptr;
    float *scratch=nullptr;
    size_t scratch_floats=0;
public:
    std::mutex call_lock;
    // Gate->Down intermediate for prefill, shared by every plan: calls serialise on
    // call_lock, so one buffer sized for the largest call replaces one per plan/layer.
    // Decode uses a small per-plan buffer instead (see Plan::decode_mid).
    float *intermediate(size_t floats) {
        if(floats>scratch_floats) {
            size_t bytes=(floats*sizeof(float)+63)/64*64;
            void *p=std::aligned_alloc(64,bytes);
            if(!p)throw std::bad_alloc();
            std::free(scratch);scratch=static_cast<float*>(p);scratch_floats=bytes/sizeof(float);
        }
        return scratch;
    }
    explicit Executor(int n):threads(n),nodes(topology().nodes),done(new Done[n]) {
        if(const char *s=std::getenv("XTIER_SPIN_COUNT")) {
            spin_limit=std::stoi(s);
            if(spin_limit<1||spin_limit>4194304)throw std::runtime_error("Invalid spin count");
        }
        spin_budget.store(spin_limit);
        if(const char *s=std::getenv("XTIER_PREFILL_SPIN_COUNT")) {
            prefill_spin=std::stoi(s);
            if(prefill_spin<1||prefill_spin>4194304)throw std::runtime_error("Invalid prefill spin count");
        }
        const int cores=node_cores(n);
        {
            static std::once_flag logged;
            std::call_once(logged,[&]{std::fprintf(stderr,"[xtier] %d worker nodes, %d threads (%d cores + %d SMT per node): %s\n",
                                                   nodes,n,std::min(cores,n/nodes),std::max(0,n/nodes-cores),topology().describe.c_str());});
        }
        try {
            for(int i=0;i<n;i++)workers.emplace_back([this,i,n,cores] {
                // Lanes past the node's first `cores` physical cores run on their SMT siblings.
                int node=i%nodes,lane=i/nodes,lanes=n/nodes,cpu=worker_cpu(i,cores);
                cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(cpu,&cpus);
                if(pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus))affinity_error.store(1);
                uint32_t seen=0;int spins=0;
                while(!quit.load(std::memory_order_relaxed)) {
                    uint32_t e=epoch.load(std::memory_order_acquire);
                    if(e==seen) {
                        _mm_pause();
                        if(++spins>=spin_budget.load(std::memory_order_relaxed)) {
                            if(epoch.load(std::memory_order_acquire)==seen && !quit.load())
                                syscall(SYS_futex,&epoch,FUTEX_WAIT_PRIVATE,seen,nullptr,nullptr,0);
                            spins=0;
                        }
                        continue;
                    }
                    if(quit.load(std::memory_order_relaxed))break;
#if XTIER_LANE_TIMING
                    lane_start[i]=now_ns();
#endif
                    spins=0;current(context,node,lane,lanes);seen=e;
#if XTIER_LANE_TIMING
                    lane_end[i]=now_ns();
#endif
#if XTIER_NODE_BARRIER
                    if(lane==0) {
                        // Gather this node's workers on local cache lines; the caller
                        // then polls one flag per node instead of one per worker.
                        for(int j=i+nodes;j<n;j+=nodes)while(done[j].epoch.load(std::memory_order_acquire)!=e)_mm_pause();
                        node_done[node].epoch.store(e,std::memory_order_release);
                        continue;
                    }
#endif
                    done[i].epoch.store(e,std::memory_order_release);
                }
            });
            dispatch(nullptr,[](void*,int,int,int){});
            if(affinity_error.load())throw std::runtime_error("CPU affinity rejected");
        } catch(...) {stop();throw;}
    }
    // Set before dispatching a call: how long workers keep spinning after it.
    void set_decode(bool decode){spin_budget.store(decode?spin_limit:std::min(spin_limit,prefill_spin),std::memory_order_relaxed);}
#if XTIER_LANE_TIMING
    static int64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    int64_t lane_start[128]{},lane_end[128]{};
    // per tag (phase): calls, sum of wake(mean start), first end, mean end, last end, all relative to dispatch
    double acc[4][6]{};
public:
    int tag=0;
    void report() {
        const char *names[4]={"init","gate","down","?"};
        for(int t=1;t<3;t++)if(acc[t][0]>0) {
            double c=acc[t][0];
            std::fprintf(stderr,"[lane-timing] %s calls=%.0f wake_mean=%.1fus first_end=%.1fus mean_end=%.1fus last_end=%.1fus caller_seen=%.1fus barrier_idle=%.1f%%\n",
                names[t],c,acc[t][1]/c/1e3,acc[t][2]/c/1e3,acc[t][3]/c/1e3,acc[t][4]/c/1e3,acc[t][5]/c/1e3,
                100.0*(acc[t][4]-acc[t][3])/acc[t][4]);
        }
    }
#endif
    void dispatch(void *object,Task fn) {
#if XTIER_LANE_TIMING
        int64_t t0=now_ns();
#endif
        context=object;current=fn;uint32_t e=epoch.load(std::memory_order_relaxed)+1;
        epoch.store(e,std::memory_order_release);
        syscall(SYS_futex,&epoch,FUTEX_WAKE_PRIVATE,threads,nullptr,nullptr,0);
#if XTIER_NODE_BARRIER
        for(int node=0;node<nodes;node++)while(node_done[node].epoch.load(std::memory_order_acquire)!=e)_mm_pause();
#else
        for(int i=0;i<threads;i++)while(done[i].epoch.load(std::memory_order_acquire)!=e)_mm_pause();
#endif
#if XTIER_LANE_TIMING
        int64_t seen=now_ns();
        if(object&&tag>0&&tag<3) {
            double ws=0,me=0,fe=1e18,le=0;
            for(int i=0;i<threads;i++){double a=lane_start[i]-t0,b=lane_end[i]-t0;ws+=a;me+=b;fe=std::min(fe,b);le=std::max(le,b);}
            auto &a=acc[tag];a[0]++;a[1]+=ws/threads;a[2]+=fe;a[3]+=me/threads;a[4]+=le;a[5]+=seen-t0;
        }
#endif
    }
    int size() const{return threads;}
    int node_count() const{return nodes;}
    void stop() {
        quit.store(true);epoch.fetch_add(1,std::memory_order_seq_cst);
        syscall(SYS_futex,&epoch,FUTEX_WAKE_PRIVATE,threads,nullptr,nullptr,0);
        for(auto &w:workers)if(w.joinable())w.join();
    }
    ~Executor(){
#if XTIER_LANE_TIMING
        report();
#endif
        stop();std::free(scratch);}
};
std::shared_ptr<Executor> get_executor(int threads) {
    static std::mutex mutex;
    static std::weak_ptr<Executor> existing;
    std::lock_guard<std::mutex> lock(mutex);
    auto pool=existing.lock();
    if(pool && pool->size()!=threads)throw std::runtime_error("Active plans must use the same thread count");
    if(!pool){pool=std::make_shared<Executor>(threads);existing=pool;}
    return pool;
}

// Per-worker private accumulation buffer (Down), grown on demand.
float *local_buffer(size_t floats) {
    static thread_local std::vector<float> buffer;
    if(buffer.size()<floats)buffer.resize(floats);
    return buffer.data();
}

// MXFP4 uses the same E2M1 nibbles, with one UE8M0 power-of-two scale per 32.
// Build only when requested, after ISA selection; existing NVFP4 tables stay constexpr.
struct MxTables {
    alignas(64) std::array<std::array<uint16_t,16>,256> bf{};
    alignas(64) std::array<std::array<float,16>,256> fp{};
    alignas(64) std::array<std::array<uint8_t,32>,256> bytes{};
    MxTables() {
        for(int s=0;s<255;s++)for(int q=0;q<16;q++) {
            uint16_t b=tobf(std::ldexp(fp4[q],s-127));
            bf[s][q]=b; fp[s][q]=__builtin_bit_cast(float,uint32_t(b)<<16);
            bytes[s][q]=uint8_t(b);bytes[s][16+q]=uint8_t(b>>8);
        }
    }
};
const MxTables& mx_tables(){static const MxTables tables;return tables;}

// One generic implementation, specialised once at plan construction. No format branch
// or variable scale stride remains in an NVFP4 weight loop.
// cold/store.cpp's reads, handed over by the dispatch library (it owns the store): xtier_cold_read (synchronous) and
// xtier_cold_submit / xtier_cold_finish (a ticket; finish waits for every job). dispatch.cpp has the same table.
using ColdRead=int(*)(void*,int,const int*,const long long*,void*const*,const size_t*,int);
using ColdSubmit=void*(*)(void*,int,const int*,const long long*,void*const*,const size_t*,int,const int*,int);
using ColdFinish=int(*)(void*,void*);
struct ColdIO {ColdRead read;ColdSubmit submit;ColdFinish finish;};
struct PlanBase {
    virtual ~PlanBase()=default;
    virtual int forward(int,int,const uint16_t*,const int32_t*,const float*,float*)=0;
    virtual void set_read_only(bool)=0;
    virtual void timings(double*)=0;
    virtual void shape(int*) const=0;
    virtual int localize()=0;
    virtual int node_count() const=0;
    virtual void shard(int,void**,size_t*) const=0;
    virtual void set_swiglu(float alpha,float beta)=0;
    virtual int place(int expert,int slot)=0;
    virtual int slot_table(int32_t *table)=0;
    virtual int slot_pieces(int slot,int node,void **pointers,size_t *bytes) const=0;
    virtual int attach_cold(void *store,const ColdIO *io,const int32_t *files,const long long *offsets,const float *scores,
                            float half_life)=0;
    virtual int pin(int expert)=0;
    virtual void unpin(int slot)=0;
    virtual void cold_stats(double *out)=0;
};
template<bool MX> class Plan: public PlanBase {
    static constexpr int scale_format=MX?1:0,scale_shift=MX?5:4;
    const std::array<uint16_t,16>* bf_table=lut.data();
#if XTIER_AVX512
    const std::array<float,16>* fp_table=flut.data();
    __m512bh unpack(const uint8_t* p,const uint8_t* s) const {if constexpr(MX)return ::unpack(p,s,bf_table,true);else return ::unpack(p,s);}
    void unpack_f32(const uint8_t* p,const uint8_t* s,__m512& a,__m512& b) const {if constexpr(MX)::unpack_f32(p,s,a,b,fp_table,true);else ::unpack_f32(p,s,a,b);}
#else
    const std::array<uint8_t,32>* byte_table=blut.data();
    void unpack32(const uint8_t* p,const uint8_t* s,__m256 w[4]) const {if constexpr(MX)::unpack32(p,s,w,byte_table,true);else ::unpack32(p,s,w);}
#endif
    void copy_scales(uint8_t* dst,const uint8_t* src,size_t count) {
        std::memcpy(dst,src,count);
    }
    int hidden,inter,experts;
    float swiglu_limit;
    // GPT-OSS SwiGLU (MiniMax M3): gate*sigmoid(alpha*gate)*(up+beta) after the clamp; 1 / 0 = the plain SiLU path
    float swiglu_alpha=1.f,swiglu_beta=0.f;
    void set_swiglu(float alpha,float beta) override {swiglu_alpha=alpha;swiglu_beta=beta;}
    std::shared_ptr<Executor> pool;
    std::vector<Shard> shards;  // one per worker node
    // RAM slots (xtier_create4, cold experts on NVMe): the shards hold `slots` experts' weights; slot_of[e] is expert
    // e's slot, -1 while it is only on NVMe. Without them slots == experts and slot e holds expert e.
    int slots=0;
    std::vector<int32_t> slot_of,expert_of;
    // Cold experts (xtier_cold_attach): every expert's record on NVMe (file, offset; file -1: it never leaves RAM). A
    // forward first reads the experts it routes to that are not in RAM (ensure_resident). resident_lock guards the
    // tables against the pins of readers on other threads; the forward itself is the only writer.
    void *cold_store=nullptr;
    ColdIO cold_io{};
    bool cold_overlap=true;           // XTIER_COLD_OVERLAP: a forward computes the experts in RAM while it reads the others
    std::vector<int> cold_taken;      // the slots the reads in flight land in
    std::vector<int32_t> cold_split;  // the routes of the two passes
    std::vector<float> cold_late;     // the second pass's output
    std::vector<int32_t> cold_file,pins;   // cold_file, cold_offset: per expert and node part
    bool stored(int e) const {return cold_file[size_t(e)*nodes]>=0;}
    std::vector<long long> cold_offset,cold_mark,cold_stamp;
    std::vector<double> cold_score;
    std::vector<int> cold_used,cold_miss;
    double cold_log=0;                       // log of the per-forward decay
    long long cold_forwards=0,cold_loads=0;
    double cold_seconds=0;
    long long cold_small_forwards=0,cold_small_loads=0;   // the forwards of at most 8 tokens (decode steps)
    double cold_small_seconds=0;
    std::mutex resident_lock;
    int nodes=4;
    std::vector<float> globals13,globals2;
    int phase=0,m=0,groups_n=0;
    const uint8_t *source13,*source2,*scale13,*scale2;
    const uint16_t *input=nullptr;
    float *output=nullptr,*mid=nullptr;
    // Decode (m<=DECODE_M) intermediate, per plan: a buffer shared across layers would
    // be rewritten by the next layer's Gate while every core still holds its lines from
    // this layer's Down, costing cross-node invalidations on every store (~2% at M=4).
    std::unique_ptr<float,decltype(&std::free)> decode_mid{nullptr,&std::free};
    std::vector<Group> groups;
    std::mutex call_lock;
    double last_times[2]{};
    bool read_only=false;  // bench probe: stream the same bytes without computing
    // XTIER_DOWN_BF16=1: the Gate epilogue rounds the intermediate to BF16 and Down uses
    // BF16 dot products (FP32 accumulation). Still finer than the FP8 activations of the
    // DeepSeek V4 reference; off by default (Qwen3.8 keeps the FP32 path bit for bit).
    bool down_bf16=false;
    // XTIER_DYNAMIC=1|2 (decode, m<=DECODE_M): workers of a node claim Gate/Down row chunks
    // from a node-local counter instead of a fixed equal share. Workers sharing a core
    // (SMT) or a busier node no longer hold the whole phase: measured 11-31% idle at the
    // phase barriers with 112 threads. Row results and the expert order of every output
    // element are unchanged, so outputs stay bit for bit identical.
    // =2 (guided): a claim takes max(grain, remaining/(2*lanes)) rows, so chunks start
    // long (contiguous weight streams for the prefetcher) and shrink towards the end.
    int dynamic=0,gate_grain=8,down_grain=4;  // smallest chunks (microkernel row multiples)
    struct alignas(64) Counter {std::atomic<int> v{0};};
    std::array<Counter,MAX_NODES> next_row;
    // Claims [begin, begin+size) of this node's `total` rows; size is a multiple of `unit`.
    int claim(int node,int total,int grain,int unit,int lanes,int &begin) {
        auto &c=next_row[node].v;
        if(dynamic==1) {
            begin=c.fetch_add(grain,std::memory_order_relaxed);
            return begin<total?std::min(grain,total-begin):0;
        }
        int q=c.load(std::memory_order_relaxed);
        while(q<total) {
            int size=std::max(grain,(total-q)/(2*lanes)/unit*unit);
            if(c.compare_exchange_weak(q,q+size,std::memory_order_relaxed)){begin=q;return std::min(size,total-q);}
        }
        return 0;
    }

    void initialize_shard(int node) {
        const size_t H=hidden,I=inter,IN=I/nodes,HN=H/nodes;auto &s=shards[node];
        for(size_t slot=0;slot<size_t(slots);slot++) {
            const size_t e=size_t(expert_of[slot]),gi=slot*IN,di=slot*HN;
            size_t srcg=e*2*I+node*IN,srcu=srcg+I;
            size_t srcd=e*H+node*HN;
            std::memcpy(s.gate.get()+gi*(H/2),source13+srcg*(H/2),IN*(H/2));
            std::memcpy(s.up.get()+gi*(H/2),source13+srcu*(H/2),IN*(H/2));
            copy_scales(s.sg.get()+gi*(H>>scale_shift),scale13+srcg*(H>>scale_shift),IN*(H>>scale_shift));
            copy_scales(s.su.get()+gi*(H>>scale_shift),scale13+srcu*(H>>scale_shift),IN*(H>>scale_shift));
            std::memcpy(s.down.get()+di*(I/2),source2+srcd*(I/2),HN*(I/2));
            copy_scales(s.sd.get()+di*(I>>scale_shift),scale2+srcd*(I>>scale_shift),HN*(I>>scale_shift));
        }
    }

    int group_expert(int g) const {return groups[g].expert;}
#if XTIER_AVX512
    // HC/IC: compile-time H and I for specialised shapes, 0 = runtime values.
    template<int M,int HC,int IC> void gate_rows(int node,int g,int begin,int end) {
        const int H=HC?HC:hidden,I=IC?IC:inter,IN=I/nodes;
        // Locals: stores into the intermediate must not force reloads of float members.
        float *__restrict const out_mid=mid;const float limit=swiglu_limit;
        const float alpha=swiglu_alpha,beta=swiglu_beta;const bool oai=alpha!=1.f||beta!=0.f;
        const float scale_gate=globals13[2*group_expert(g)],scale_up=globals13[2*group_expert(g)+1];
        const auto &s=shards[node];const auto &group=groups[g];const size_t slot=size_t(group.slot);
        const uint16_t *xs[M];for(int t=0;t<M;t++)xs[t]=input+size_t(group.token[t])*H;
        // Rows per iteration: 2 up to M=4 (and M<=XTIER_WIDE_ROWS_M with 2x2xM accumulators
        // still in registers), 1 beyond. Row results do not depend on R.
        constexpr int R=(M<=4||M<=XTIER_WIDE_ROWS_M)?2:1;
        for(int r=begin;r<end;r+=R) {
            __m512 ag[R][M],au[R][M];
            for(int j=0;j<R;j++)for(int t=0;t<M;t++){ag[j][t]=_mm512_setzero_ps();au[j][t]=_mm512_setzero_ps();}
            for(int c=0;c<H;c+=32) {
                __m512bh x[M];for(int t=0;t<M;t++)x[t]=(__m512bh)_mm512_loadu_si512(xs[t]+c);
                for(int j=0;j<R;j++) {
                    size_t base=slot*IN+r+j;
                    __m512bh a=unpack(s.gate.get()+base*(H/2)+c/2,s.sg.get()+base*(H>>scale_shift)+(c>>scale_shift));
                    __m512bh b=unpack(s.up.get()+base*(H/2)+c/2,s.su.get()+base*(H>>scale_shift)+(c>>scale_shift));
                    for(int t=0;t<M;t++) {
                        ag[j][t]=_mm512_dpbf16_ps(ag[j][t],a,x[t]);au[j][t]=_mm512_dpbf16_ps(au[j][t],b,x[t]);
                    }
                }
            }
            for(int j=0;j<R;j++)for(int t=0;t<M;t++) {
                float gate=_mm512_reduce_add_ps(ag[j][t])*scale_gate;
                float up=_mm512_reduce_add_ps(au[j][t])*scale_up;
                if(limit>0.f) {
                    // Clamped SwiGLU (DeepSeek V4 Expert): up to [-L,L], gate to (-inf,L].
                    up=std::min(std::max(up,-limit),limit);
                    gate=std::min(gate,limit);
                }
                float value=oai?(gate/(1.f+std::exp(-alpha*gate)))*(up+beta):(gate/(1.f+std::exp(-gate)))*up;
                size_t at=(size_t(group.row)+t)*I+node*IN+r+j;
                if(down_bf16)reinterpret_cast<uint16_t*>(out_mid)[at]=tobf(value);
                else out_mid[at]=value;
            }
        }
    }

    template<int M,int HC,int IC> void down_rows_bf16(int node,int g,int begin,int end,float *destination) {
        const int H=HC?HC:hidden,I=IC?IC:inter,HN=H/nodes;
        const auto &s=shards[node];const auto &group=groups[g];int e=group.expert;const size_t slot=size_t(group.slot);
        const uint16_t *__restrict const in_mid=reinterpret_cast<const uint16_t*>(mid);const float scale=globals2[e];
        float *__restrict const out=destination?destination:output;
        float route[M];int token[M];for(int t=0;t<M;t++){route[t]=group.route[t];token[t]=group.token[t];}
        constexpr int R=M<=4?XTIER_DOWN_R4:(M<=XTIER_WIDE_ROWS_M?2:1);
        for(int r=begin;r<end;r+=R) {
            __m512 acc[R][M];for(int j=0;j<R;j++)for(int t=0;t<M;t++)acc[j][t]=_mm512_setzero_ps();
            for(int c=0;c<I;c+=32) {
                __m512bh x[M];
                for(int t=0;t<M;t++)x[t]=(__m512bh)_mm512_loadu_si512(in_mid+(size_t(group.row)+t)*I+c);
                for(int j=0;j<R;j++) {
                    size_t base=slot*HN+r+j;
                    __m512bh w=unpack(s.down.get()+base*(I/2)+c/2,s.sd.get()+base*(I>>scale_shift)+(c>>scale_shift));
                    for(int t=0;t<M;t++)acc[j][t]=_mm512_dpbf16_ps(acc[j][t],w,x[t]);
                }
            }
            for(int j=0;j<R;j++)for(int t=0;t<M;t++) {
                float val=_mm512_reduce_add_ps(acc[j][t])*scale;
                out[size_t(token[t])*H+node*HN+r+j]+=val*route[t];
            }
        }
    }

    template<int M,int HC,int IC> void down_rows(int node,int g,int begin,int end,float *destination=nullptr) {
        if(down_bf16){down_rows_bf16<M,HC,IC>(node,g,begin,end,destination);return;}
        const int H=HC?HC:hidden,I=IC?IC:inter,HN=H/nodes;
        const auto &s=shards[node];const auto &group=groups[g];int e=group.expert;const size_t slot=size_t(group.slot);
        const float *__restrict const in_mid=mid;const float scale=globals2[e];
        float *__restrict const out=destination?destination:output;
        float route[M];int token[M];for(int t=0;t<M;t++){route[t]=group.route[t];token[t]=group.token[t];}
        constexpr int R=M<=4?XTIER_DOWN_R4:(M<=XTIER_WIDE_ROWS_M?2:1);
        for(int r=begin;r<end;r+=R) {
            __m512 acc[R][M];for(int j=0;j<R;j++)for(int t=0;t<M;t++)acc[j][t]=_mm512_setzero_ps();
            for(int c=0;c<I;c+=32) {
                __m512 xl[M],xh[M];for(int t=0;t<M;t++) {
                    const float *row=in_mid+(size_t(group.row)+t)*I;
                    xl[t]=_mm512_loadu_ps(row+c);xh[t]=_mm512_loadu_ps(row+c+16);
                }
                for(int j=0;j<R;j++) {
                    size_t base=slot*HN+r+j;
                    __m512 lo,hi;
#if XTIER_DOWN_F32_LUT
                    unpack_f32(s.down.get()+base*(I/2)+c/2,s.sd.get()+base*(I>>scale_shift)+(c>>scale_shift),lo,hi);
#else
                    asfloat(unpack(s.down.get()+base*(I/2)+c/2,s.sd.get()+base*(I>>scale_shift)+(c>>scale_shift)),lo,hi);
#endif
                    for(int t=0;t<M;t++) {
                        acc[j][t]=_mm512_fmadd_ps(lo,xl[t],acc[j][t]);
                        acc[j][t]=_mm512_fmadd_ps(hi,xh[t],acc[j][t]);
                    }
                }
            }
            for(int j=0;j<R;j++)for(int t=0;t<M;t++) {
                float val=_mm512_reduce_add_ps(acc[j][t])*scale;
                out[size_t(token[t])*H+node*HN+r+j]+=val*route[t];
            }
        }
    }
#else
    // AVX2 (16 YMM registers): FP32 weights from the table, BF16 inputs widened, FMA into
    // FP32. Products are exact (<=6 significant weight bits x 8 input bits), so only the
    // summation order differs from the AVX512 BF16 dot products. R rows x MT tokens
    // (tokens t0.. of the group) per iteration; S=2 keeps the low and high eight weights
    // of a block in separate accumulators (two FMA chains) where registers allow.
    template<int R,int MT,int S,int HC,int IC> void gate_core(int node,int g,int begin,int end,int t0) {
        static_assert(R==1||MT==1,"two rows per iteration only for one token");
        const int H=HC?HC:hidden,I=IC?IC:inter,IN=I/nodes;
        float *__restrict const out_mid=mid;const float limit=swiglu_limit;const bool round=down_bf16;
        const float alpha=swiglu_alpha,beta=swiglu_beta;const bool oai=alpha!=1.f||beta!=0.f;
        const float scale_gate=globals13[2*group_expert(g)],scale_up=globals13[2*group_expert(g)+1];
        const auto &s=shards[node];const auto &group=groups[g];const size_t slot=size_t(group.slot);
        const uint16_t *x1=input+size_t(group.token[t0])*H;
        // Several tokens: FP32 inputs once per call (thread-local), read by the FMAs from cache,
        // so the registers hold one row's weights and the accumulators only.
        const float *xf=nullptr;
        if constexpr(MT>1) {
            static thread_local std::vector<float> buffer;
            if(buffer.size()<size_t(MT)*H)buffer.resize(size_t(MT)*H);
            for(int t=0;t<MT;t++) {
                const uint16_t *src=input+size_t(group.token[t0+t])*H;float *dst=buffer.data()+size_t(t)*H;
                for(int c=0;c<H;c+=16){__m256 lo,hi;bf16x16(src+c,lo,hi);_mm256_storeu_ps(dst+c,lo);_mm256_storeu_ps(dst+c+8,hi);}
            }
            xf=buffer.data();
        }
        for(int r=begin;r<end;r+=R) {
            __m256 ag[R*MT*S],au[R*MT*S];
            for(int q=0;q<R*MT*S;q++){ag[q]=_mm256_setzero_ps();au[q]=_mm256_setzero_ps();}
            const uint8_t *gw[R],*uw[R],*gs[R],*us[R];
            for(int j=0;j<R;j++) {
                size_t base=slot*IN+r+j;
                gw[j]=s.gate.get()+base*(H/2);uw[j]=s.up.get()+base*(H/2);
                gs[j]=s.sg.get()+base*(H>>scale_shift);us[j]=s.su.get()+base*(H>>scale_shift);
            }
            for(int c=0;c<H;c+=32) {
                __m256 x[4];
                if constexpr(MT==1){bf16x16(x1+c,x[0],x[1]);bf16x16(x1+c+16,x[2],x[3]);}
                for(int j=0;j<R;j++) {
                    // Per accumulator the FMAs run block by block, low then high eight weights.
                    __m256 w[4];
                    unpack32(gw[j]+c/2,gs[j]+(c>>scale_shift),w);
                    for(int t=0;t<MT;t++) {
                        __m256 *ga=ag+(j*MT+t)*S;
                        for(int k=0;k<4;k++) {
                            __m256 xk;
                            if constexpr(MT==1)xk=x[k];else xk=_mm256_loadu_ps(xf+size_t(t)*H+c+8*k);
                            ga[(k&1)*(S-1)]=_mm256_fmadd_ps(w[k],xk,ga[(k&1)*(S-1)]);
                        }
                    }
                    unpack32(uw[j]+c/2,us[j]+(c>>scale_shift),w);
                    for(int t=0;t<MT;t++) {
                        __m256 *ua=au+(j*MT+t)*S;
                        for(int k=0;k<4;k++) {
                            __m256 xk;
                            if constexpr(MT==1)xk=x[k];else xk=_mm256_loadu_ps(xf+size_t(t)*H+c+8*k);
                            ua[(k&1)*(S-1)]=_mm256_fmadd_ps(w[k],xk,ua[(k&1)*(S-1)]);
                        }
                    }
                }
            }
            for(int j=0;j<R;j++)for(int t=0;t<MT;t++) {
                const __m256 *ga=ag+(j*MT+t)*S,*ua=au+(j*MT+t)*S;
                float gate=hsum(S==2?_mm256_add_ps(ga[0],ga[1]):ga[0])*scale_gate;
                float up=hsum(S==2?_mm256_add_ps(ua[0],ua[1]):ua[0])*scale_up;
                if(limit>0.f) {
                    up=std::min(std::max(up,-limit),limit);
                    gate=std::min(gate,limit);
                }
                float value=oai?(gate/(1.f+std::exp(-alpha*gate)))*(up+beta):(gate/(1.f+std::exp(-gate)))*up;
                // The intermediate stays FP32 here; XTIER_DOWN_BF16 rounds its values only.
                out_mid[(size_t(group.row)+t0+t)*I+node*IN+r+j]=round?round_bf16(value):value;
            }
        }
    }
    template<int M,int HC,int IC> void gate_rows(int node,int g,int begin,int end) {
        if constexpr(M==1)gate_core<2,1,2,HC,IC>(node,g,begin,end,0);
        else if constexpr(M<=4)gate_core<1,M,1,HC,IC>(node,g,begin,end,0);
        else {gate_core<1,4,1,HC,IC>(node,g,begin,end,0);gate_core<1,M-4,1,HC,IC>(node,g,begin,end,4);}
    }
    template<int R,int MT,int S,int HC,int IC> void down_core(int node,int g,int begin,int end,float *__restrict out,int t0) {
        const int H=HC?HC:hidden,I=IC?IC:inter,HN=H/nodes;
        const auto &s=shards[node];const auto &group=groups[g];int e=group.expert;const size_t slot=size_t(group.slot);
        const float scale=globals2[e];
        float route[MT];int token[MT];const float *rows[MT];
        for(int t=0;t<MT;t++){route[t]=group.route[t0+t];token[t]=group.token[t0+t];rows[t]=mid+(size_t(group.row)+t0+t)*I;}
        for(int r=begin;r<end;r+=R) {
            __m256 acc[R*MT*S];for(int q=0;q<R*MT*S;q++)acc[q]=_mm256_setzero_ps();
            const uint8_t *dw[R],*ds[R];
            for(int j=0;j<R;j++){size_t base=slot*HN+r+j;dw[j]=s.down.get()+base*(I/2);ds[j]=s.sd.get()+base*(I>>scale_shift);}
            for(int c=0;c<I;c+=32) {
                for(int j=0;j<R;j++) {
                    __m256 w[4];unpack32(dw[j]+c/2,ds[j]+(c>>scale_shift),w);
                    for(int t=0;t<MT;t++) {
                        __m256 *a=acc+(j*MT+t)*S;
                        for(int k=0;k<4;k++)a[(k&1)*(S-1)]=_mm256_fmadd_ps(w[k],_mm256_loadu_ps(rows[t]+c+8*k),a[(k&1)*(S-1)]);
                    }
                }
            }
            for(int j=0;j<R;j++)for(int t=0;t<MT;t++) {
                const __m256 *a=acc+(j*MT+t)*S;
                float val=hsum(S==2?_mm256_add_ps(a[0],a[1]):a[0])*scale;
                out[size_t(token[t])*H+node*HN+r+j]+=val*route[t];
            }
        }
    }
    // Rows per iteration divide the four-row split boundaries, as in the AVX512 kernel.
    template<int M,int HC,int IC> void down_rows(int node,int g,int begin,int end,float *destination=nullptr) {
        float *out=destination?destination:output;
        if constexpr(M==1)down_core<4,1,2,HC,IC>(node,g,begin,end,out,0);
        else if constexpr(M<=3)down_core<4,M,1,HC,IC>(node,g,begin,end,out,0);
        else if constexpr(M==4)down_core<2,4,1,HC,IC>(node,g,begin,end,out,0);
        else down_core<1,M,1,HC,IC>(node,g,begin,end,out,0);
    }
#endif

    template<int HC,int IC> void gate_any(int count,int node,int g,int begin,int end) {
        switch(count) {
            case 1:gate_rows<1,HC,IC>(node,g,begin,end);break;
            case 2:gate_rows<2,HC,IC>(node,g,begin,end);break;
            case 3:gate_rows<3,HC,IC>(node,g,begin,end);break;
            case 4:gate_rows<4,HC,IC>(node,g,begin,end);break;
#if XTIER_PREFILL_MICRO_TOKENS == 8
            case 5:gate_rows<5,HC,IC>(node,g,begin,end);break;
            case 6:gate_rows<6,HC,IC>(node,g,begin,end);break;
            case 7:gate_rows<7,HC,IC>(node,g,begin,end);break;
            case 8:gate_rows<8,HC,IC>(node,g,begin,end);break;
#endif
        }
    }
    template<int HC,int IC> void down_any(int count,int node,int g,int begin,int end,float *destination=nullptr) {
        switch(count) {
            case 1:down_rows<1,HC,IC>(node,g,begin,end,destination);break;
            case 2:down_rows<2,HC,IC>(node,g,begin,end,destination);break;
            case 3:down_rows<3,HC,IC>(node,g,begin,end,destination);break;
            case 4:down_rows<4,HC,IC>(node,g,begin,end,destination);break;
#if XTIER_PREFILL_MICRO_TOKENS == 8
            case 5:down_rows<5,HC,IC>(node,g,begin,end,destination);break;
            case 6:down_rows<6,HC,IC>(node,g,begin,end,destination);break;
            case 7:down_rows<7,HC,IC>(node,g,begin,end,destination);break;
            case 8:down_rows<8,HC,IC>(node,g,begin,end,destination);break;
#endif
        }
    }

    template<int HC,int IC> void run(int node,int lane,int lanes) {
        const int H=HC?HC:hidden,I=IC?IC:inter,IN=I/nodes,HN=H/nodes;
        if(phase==1) {
            if(dynamic&&m<=DECODE_M) {
                const int total=groups_n*IN;
                for(int q,size;(size=claim(node,total,gate_grain,2,lanes,q))>0;) {
                    int stop=q+size;
                    while(q<stop) {
                        int g=q/IN,begin=q%IN,end=std::min(IN,begin+(stop-q));
                        gate_any<HC,IC>(groups[g].count,node,g,begin,end);
                        q+=end-begin;
                    }
                }
                return;
            }
#if XTIER_BALANCED_GATE
            if(m<=DECODE_M) {
                // One contiguous span of the node's groups_n*IN Gate rows per worker,
                // split on the two-row microkernel boundary; spans may cross experts.
                // Rows are weighted by XTIER_GATE_COST_A + tokens, since a row shared
                // by several tokens costs more than a single-token row (mixed MTP routes).
                auto row_at=[&](long position) {
                    long cost=0;
                    for(int g=0;g<groups_n;g++) {
                        long unit=XTIER_GATE_COST_A+groups[g].count,span=long(IN)*unit;
                        if(position<cost+span) {
                            long r=(position-cost+unit-1)/unit;r+=r&1;
                            return g*IN+int(std::min<long>(r,IN));
                        }
                        cost+=span;
                    }
                    return groups_n*IN;
                };
                long total=0;
                for(int g=0;g<groups_n;g++)total+=long(IN)*(XTIER_GATE_COST_A+groups[g].count);
                int q=row_at(total*lane/lanes),stop=row_at(total*(lane+1)/lanes);
                while(q<stop) {
                    int g=q/IN,begin=q%IN,end=std::min(IN,begin+(stop-q));
                    gate_any<HC,IC>(groups[g].count,node,g,begin,end);
                    q+=end-begin;
                }
                return;
            }
#endif
            const int grain=16,per_group=IN/grain;
            for(int j=lane;j<groups_n*per_group;j+=lanes) {
                int g=j/per_group,begin=(j%per_group)*grain,end=begin+grain;
                gate_any<HC,IC>(groups[g].count,node,g,begin,end);
            }
        } else {
            if(dynamic&&m<=DECODE_M) {
                // Chunks keep the four-row microkernel boundaries and the private
                // accumulation of the static split below.
                float *local=local_buffer(size_t(m)*H);
                const int offset=node*HN;
                for(int begin,size;(size=claim(node,HN,down_grain,4,lanes,begin))>0;) {
                    int end=begin+size;
                    for(int t=0;t<m;t++)std::fill(local+size_t(t)*H+offset+begin,local+size_t(t)*H+offset+end,0.f);
                    if(m==1)for(int g=0;g<groups_n;g++)down_rows<1,HC,IC>(node,g,begin,end,local);
                    else for(int g=0;g<groups_n;g++)down_any<HC,IC>(groups[g].count,node,g,begin,end,local);
                    for(int t=0;t<m;t++)
                        std::memcpy(output+size_t(t)*H+offset+begin,local+size_t(t)*H+offset+begin,(end-begin)*sizeof(float));
                }
                return;
            }
            if(m==1) {
                // Four-row boundaries match the decode microkernel. Each worker
                // owns a contiguous, nearly equal span, with private accumulation.
                int begin=(lane*(HN/4)/lanes)*4;
                int end=((lane+1)*(HN/4)/lanes)*4;
                float *local=local_buffer(H);
                int offset=node*HN;
                std::fill(local+offset+begin,local+offset+end,0.f);
                for(int g=0;g<groups_n;g++)down_rows<1,HC,IC>(node,g,begin,end,local);
                std::memcpy(output+offset+begin,local+offset+begin,(end-begin)*sizeof(float));
                return;
            }
#if XTIER_BALANCED_SMALL_DOWN
            if(m<=DECODE_M) {
                // Same row split and private accumulation as M=1: equal rows per worker
                // instead of whole 16-row grains; expert order per output element unchanged.
                int begin=(lane*(HN/4)/lanes)*4;
                int end=((lane+1)*(HN/4)/lanes)*4;
                float *local=local_buffer(size_t(m)*H);
                int offset=node*HN;
                for(int t=0;t<m;t++)std::fill(local+size_t(t)*H+offset+begin,local+size_t(t)*H+offset+end,0.f);
                for(int g=0;g<groups_n;g++)down_any<HC,IC>(groups[g].count,node,g,begin,end,local);
                for(int t=0;t<m;t++)
                    std::memcpy(output+size_t(t)*H+offset+begin,local+size_t(t)*H+offset+begin,(end-begin)*sizeof(float));
                return;
            }
#endif
            const int grain=16;
            for(int j=lane;j<HN/grain;j+=lanes) {
                int begin=j*grain,end=begin+grain;
                for(int t=0;t<m;t++)std::fill(output+size_t(t)*H+node*HN+begin,output+size_t(t)*H+node*HN+end,0.f);
                for(int g=0;g<groups_n;g++)down_any<HC,IC>(groups[g].count,node,g,begin,end);
            }
        }
    }

    // Bench probe (xtier_read_probe): the Gate and Down phases stream exactly the weight
    // and scale bytes of the routed experts, split like the balanced decode rows, with no
    // arithmetic. Gives the memory ceiling for the compute kernels on this machine.
    void read_rows(int node,int lane,int lanes) {
        const size_t H=hidden,I=inter;const auto &s=shards[node];
#if XTIER_AVX512
        __m512i acc=_mm512_setzero_si512();
        auto stream=[&](const uint8_t *p,size_t n){
            size_t o=0;for(;n-o>=64;o+=64)acc=_mm512_xor_si512(acc,_mm512_loadu_si512(p+o));
            // Compact MXFP4 scale rows need not be a cache-line multiple.
            if(o<n){alignas(64) uint8_t tail[64]{};std::memcpy(tail,p+o,n-o);acc=_mm512_xor_si512(acc,_mm512_load_si512(tail));}
        };
#else
        __m256i acc=_mm256_setzero_si256();
        auto stream=[&](const uint8_t *p,size_t n){
            size_t o=0;for(;n-o>=32;o+=32)acc=_mm256_xor_si256(acc,_mm256_loadu_si256((const __m256i*)(p+o)));
            if(o<n){alignas(32) uint8_t tail[32]{};std::memcpy(tail,p+o,n-o);acc=_mm256_xor_si256(acc,_mm256_load_si256((const __m256i*)tail));}
        };
#endif
        size_t rows=phase==1?I/nodes:H/nodes,begin=rows*lane/lanes,end=rows*(lane+1)/lanes;
        for(int g=0;g<groups_n;g++) {
            size_t e=groups[g].slot;
            if(phase==1) {
                stream(s.gate.get()+(e*rows+begin)*(H/2),(end-begin)*(H/2));
                stream(s.up.get()+(e*rows+begin)*(H/2),(end-begin)*(H/2));
                stream(s.sg.get()+(e*rows+begin)*(H>>scale_shift),(end-begin)*(H>>scale_shift));
                stream(s.su.get()+(e*rows+begin)*(H>>scale_shift),(end-begin)*(H>>scale_shift));
            } else {
                stream(s.down.get()+(e*rows+begin)*(I/2),(end-begin)*(I/2));
                stream(s.sd.get()+(e*rows+begin)*(I>>scale_shift),(end-begin)*(I>>scale_shift));
            }
        }
#if XTIER_AVX512
        static volatile long long sink;sink=_mm512_reduce_add_epi64(acc);
#else
        static volatile long long sink;sink=_mm256_extract_epi64(acc,0)+_mm256_extract_epi64(acc,3);
#endif
    }
    void task(int node,int lane,int lanes) {
        if(phase==0) {if(lane==0)initialize_shard(node);return;}
        if(read_only){read_rows(node,lane,lanes);return;}
        if(hidden==2560&&inter==640)run<2560,640>(node,lane,lanes);        // Qwen3.8 Flash Next
        else if(hidden==5120&&inter==2304)run<5120,2304>(node,lane,lanes); // DeepSeek V4.1 Flash
        else if(hidden==4096&&inter==2048)run<4096,2048>(node,lane,lanes); // GLM-5.3 Flash
        else run<0,0>(node,lane,lanes);
    }

    void dispatch(int p) {
        for(auto &c:next_row)c.v.store(0,std::memory_order_relaxed);
#if XTIER_LANE_TIMING
        pool->tag=p;
#endif
        phase=p;pool->dispatch(this,[](void *self,int node,int lane,int lanes){
            static_cast<Plan*>(self)->task(node,lane,lanes);
        });
    }

public:
    Plan(int h,int i,int e,int n,float limit,const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,
         const uint8_t *s2,const float *g13,const float *g2,int format=0,int slot_count=0,const int32_t *resident=nullptr)
        :hidden(h),inter(i),experts(e),swiglu_limit(limit),pool(get_executor(n)),globals13(g13,g13+e*2),
         globals2(g2,g2+e),source13(w13),source2(w2),scale13(s13),scale2(s2),groups(MAX_ROUTES) {
        if(format<0||format>1)throw std::runtime_error("Unknown scale format");
        if(format) {
            bf_table=mx_tables().bf.data();
#if XTIER_AVX512
            fp_table=mx_tables().fp.data();
#else
            byte_table=mx_tables().bytes.data();
#endif
        }
        if(e<10 || e>MAX_EXPERTS)throw std::runtime_error("E=10..512");
        slots=slot_count>0?slot_count:e;
        if(slots>e)throw std::runtime_error("More RAM slots than experts");
        slot_of.assign(e,-1);expert_of.assign(slots,-1);
        for(int j=0;j<slots;j++) {
            const int x=resident?resident[j]:j;
            if(x<0||x>=e||slot_of[x]>=0)throw std::runtime_error("Invalid resident experts");
            slot_of[x]=j;expert_of[j]=x;
        }
        if(h<64 || h>32768 || h%64 || i<64 || i>32768 || i%64)throw std::runtime_error("hidden and intermediate must be multiples of 64");
        nodes=pool->node_count();
        // Every node's share of the I and H rows must stay a multiple of the 16-row grains.
        if(i%(16*nodes) || h%(16*nodes))
            throw std::runtime_error("hidden "+std::to_string(h)+" / intermediate "+std::to_string(i)+" cannot be split over "+
                                     std::to_string(nodes)+" worker nodes (need multiples of "+std::to_string(16*nodes)+")");
        shards.resize(nodes);
        if(!std::isfinite(limit) || limit<0)throw std::runtime_error("Invalid SwiGLU limit");
        for(float g:globals13)if(!std::isfinite(g)||g<=0)throw std::runtime_error("Invalid global scale");
        for(float g:globals2)if(!std::isfinite(g)||g<=0)throw std::runtime_error("Invalid global scale");
        const size_t E=e,H=h,I=i;
        for(size_t j=0;j<E*2*I*H/(format?32:16);j++)if(s13[j]>(format?254:126))throw std::runtime_error("Unsupported block scale");
        for(size_t j=0;j<E*H*I/(format?32:16);j++)if(s2[j]>(format?254:126))throw std::runtime_error("Unsupported block scale");
        const size_t IN=I/nodes,HN=H/nodes,S=slots;
        for(int node=0;node<nodes;node++) {
            auto &s=shards[node];const int mem=topology().memory[node];
            s.gate=alloc(S*IN*(H/2),mem);s.up=alloc(S*IN*(H/2),mem);
            s.sg=alloc(S*IN*(H>>scale_shift),mem);s.su=alloc(S*IN*(H>>scale_shift),mem);
            s.down=alloc(S*HN*(I/2),mem);s.sd=alloc(S*HN*(I>>scale_shift),mem);
        }
        if(const char *v=std::getenv("XTIER_DOWN_BF16"))down_bf16=std::string(v)=="1";
        if(const char *v=std::getenv("XTIER_DYNAMIC"))dynamic=std::stoi(v);
        if(dynamic<0||dynamic>2)throw std::runtime_error("XTIER_DYNAMIC must be 0, 1 or 2");
        if(const char *v=std::getenv("XTIER_GATE_GRAIN"))gate_grain=std::stoi(v);
        if(const char *v=std::getenv("XTIER_DOWN_GRAIN"))down_grain=std::stoi(v);
        if(gate_grain<2||gate_grain%2||down_grain<4||down_grain%4)
            throw std::runtime_error("XTIER_GATE_GRAIN must be even, XTIER_DOWN_GRAIN a multiple of 4");
        size_t decode_bytes=(size_t(DECODE_M)*MAX_K*I*sizeof(float)+63)/64*64;
        decode_mid.reset(static_cast<float*>(std::aligned_alloc(64,decode_bytes)));
        if(!decode_mid)throw std::bad_alloc();
        std::memset(decode_mid.get(),0,decode_bytes);
        std::lock_guard<std::mutex> lock(pool->call_lock);dispatch(0);
    }
    int forward(int tokens,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
        std::unique_lock<std::mutex> guard(call_lock,std::try_to_lock);
        if(!guard.owns_lock())throw std::runtime_error("Plan does not permit concurrent callers");
        if(tokens<1||tokens>MAX_BATCH||k<1||k>MAX_K||!x||!ids||!rw||!y)throw std::runtime_error("Invalid input");
        void *reads=cold_store?ensure_resident(tokens,k,ids):nullptr;
        if(!reads)return compute(tokens,k,x,ids,rw,y);
        // Experts being read (overlap): first the routes to the experts in RAM, then the others once their reads
        // landed, into a second buffer added to the first (two FP32 sums: not bitwise the one-pass order)
        const size_t n=size_t(tokens)*k,rows=size_t(tokens)*hidden;
        cold_split.resize(2*n);
        int32_t *early=cold_split.data(),*late=early+n;
        bool any=false;
        for(size_t j=0;j<n;j++) {
            const int e=ids[j];const bool wait=e>=0&&e<experts&&slot_of[e]<0;
            early[j]=wait?-1:e;late[j]=wait?e:-1;any|=!wait&&e>=0;
        }
        if(!any){finish_reads(reads,tokens);return compute(tokens,k,x,late,rw,y);}
        int first=0;
        try{first=compute(tokens,k,x,early,rw,y);}
        catch(...){try{finish_reads(reads,tokens);}catch(...){}throw;}
        const double phases[2]={last_times[0],last_times[1]};
        finish_reads(reads,tokens);
        cold_late.resize(rows);
        const int second=compute(tokens,k,x,late,rw,cold_late.data());
        for(size_t j=0;j<rows;j++)y[j]+=cold_late[j];
        last_times[0]+=phases[0];last_times[1]+=phases[1];
        return first+second;
    }
    // One pass over the routes (-1: skipped) into y.
    int compute(int tokens,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
        m=tokens;input=x;output=y;groups_n=0;
        std::array<int,MAX_EXPERTS> last_group;
        if(m>4)last_group.fill(-1);
        for(int t=0;t<m;t++)for(int q=0;q<k;q++) {
            int e=ids[t*k+q];float weight=rw[t*k+q];
            if(e==-1)continue;  // route computed elsewhere (hybrid CPU/GPU prefill)
            if(e<0||e>=experts||!std::isfinite(weight))throw std::runtime_error("Invalid route");
            int g=0;
            if(m<=4) {
                for(;g<groups_n;g++)if(groups[g].expert==e)break;
            } else {
                g=last_group[e];
                if(g<0 || (groups[g].count==MAX_M && groups[g].token[MAX_M-1]!=t))g=groups_n;
            }
            if(g==groups_n) {
                if(slot_of[e]<0)throw std::runtime_error("Expert "+std::to_string(e)+" is not in RAM");
                groups[g]=Group{};groups[g].expert=e;groups[g].slot=slot_of[e];groups_n++;
            }
            if(m>4)last_group[e]=g;
            auto &group=groups[g];int j=0;for(;j<group.count;j++)if(group.token[j]==t)break;
            if(j==group.count){group.token[j]=t;group.route[j]=weight;group.count++;}
            else group.route[j]+=weight;
        }
        // Same-expert subgroups are adjacent for weight-cache reuse. Keep the
        // existing small-decode ordering and arithmetic completely unchanged.
        if(m>4)std::stable_sort(groups.begin(),groups.begin()+groups_n,
                              [](const Group&a,const Group&b){return a.expert<b.expert;});
        int rows=0;
        for(int g=0;g<groups_n;g++){groups[g].row=rows;rows+=groups[g].count;}
        std::lock_guard<std::mutex> shared_lock(pool->call_lock);
        mid=m<=DECODE_M?decode_mid.get():pool->intermediate(size_t(rows)*inter);
        pool->set_decode(m<=DECODE_M);
        auto a=std::chrono::steady_clock::now();dispatch(1);
        auto b=std::chrono::steady_clock::now();dispatch(2);
        auto c=std::chrono::steady_clock::now();
        last_times[0]=std::chrono::duration<double,std::micro>(b-a).count();
        last_times[1]=std::chrono::duration<double,std::micro>(c-b).count();
        return groups_n;
    }
    void set_read_only(bool v){read_only=v;}
    void timings(double *out) {out[0]=last_times[0];out[1]=last_times[1];}
    void shape(int *out) const {out[0]=hidden;out[1]=inter;out[2]=experts;out[3]=DECODE_M;}
    // Move shard pages that landed on another node (MPOL_PREFERRED under memory pressure
    // while the model loads) back to their own node, then restore the preferred policy.
    // Call once after loading, before the shards are pinned for GPU upload.
    // Returns the number of ranges that could not be fully moved.
    int localize() {
        int failed=0;
        for(int node=0;node<nodes;node++) {
            auto &s=shards[node];
            Bytes *all[6]={&s.gate,&s.up,&s.sg,&s.su,&s.down,&s.sd};
            unsigned long mask=1ul<<topology().memory[node];
            for(auto *b:all) {
                void *p=b->get();size_t n=b->get_deleter().size;
                if(!p||!n)continue;
                if(syscall(SYS_mbind,p,n,MPOL_BIND,&mask,sizeof(mask)*8,MPOL_MF_MOVE))failed++;
                syscall(SYS_mbind,p,n,MPOL_PREFERRED,&mask,sizeof(mask)*8,0);
            }
        }
        return failed;
    }
    int node_count() const {return nodes;}
    // Cold experts: the experts this call routes to that are not in RAM are read (one job per node part, all at once)
    // into the slots of the experts with the lowest decayed use counts that this call does not use, nobody pinned and
    // that have a record on NVMe. A reserved slot (expert_of -2) belongs to no expert until its read lands: it can be
    // neither pinned nor chosen again for another expert of the same call. With cold_overlap the reads are only
    // started: returns their ticket (finish_reads); else returns nullptr once they landed.
    void *ensure_resident(int tokens,int k,const int32_t *ids) {
        const long long now=++cold_forwards;
        if(tokens<=8)cold_small_forwards++;
        cold_used.clear();
        for(int j=0;j<tokens*k;j++) {
            const int e=ids[j];
            if(e<0||e>=experts||cold_mark[e]==now)continue;   // invalid routes fail in the grouping below
            cold_mark[e]=now;cold_used.push_back(e);
        }
        for(int e:cold_used){cold_score[e]=cold_score[e]*std::exp(cold_log*double(now-cold_stamp[e]))+1;cold_stamp[e]=now;}
        cold_miss.clear();
        for(int e:cold_used)if(slot_of[e]<0)cold_miss.push_back(e);
        if(cold_miss.empty())return nullptr;
        const auto t0=std::chrono::steady_clock::now();
        std::vector<int> taken;
        {
            std::lock_guard<std::mutex> g(resident_lock);
            for(int e:cold_miss) {
                if(!stored(e))throw std::runtime_error("Expert "+std::to_string(e)+" is neither in RAM nor on NVMe");
                int best=-1;double low=0;
                for(int s=0;s<slots;s++) {
                    if(pins[s])continue;
                    const int v=expert_of[s];double u=-1;   // a free slot first
                    if(v==-2)continue;                      // reserved for another expert of this call
                    if(v>=0) {
                        if(cold_mark[v]==now||!stored(v))continue;
                        u=cold_score[v]*std::exp(cold_log*double(now-cold_stamp[v]));
                    }
                    if(best<0||u<low){best=s;low=u;}
                }
                if(best<0)throw std::runtime_error("No RAM slot for expert "+std::to_string(e));
                if(expert_of[best]>=0)slot_of[expert_of[best]]=-1;
                expert_of[best]=-2;
                taken.push_back(best);
            }
        }
        const size_t H=hidden,I=inter,IN=I/nodes,HN=H/nodes;
        const size_t sizes[6]={IN*(H/2),IN*(H/2),IN*(H>>scale_shift),IN*(H>>scale_shift),HN*(I/2),HN*(I>>scale_shift)};
        const int jobs=int(cold_miss.size())*nodes;
        std::vector<int> files(jobs);std::vector<long long> offsets(jobs);
        std::vector<void*> dst(size_t(jobs)*6);std::vector<size_t> len(size_t(jobs)*6);
        for(size_t q=0;q<cold_miss.size();q++)for(int node=0;node<nodes;node++) {
            const size_t j=q*nodes+node,s=size_t(taken[q]);const int e=cold_miss[q];const auto &sh=shards[node];
            uint8_t *base[6]={sh.gate.get(),sh.up.get(),sh.sg.get(),sh.su.get(),sh.down.get(),sh.sd.get()};
            files[j]=cold_file[size_t(e)*nodes+node];offsets[j]=cold_offset[size_t(e)*nodes+node];
            for(int p=0;p<6;p++){dst[j*6+p]=base[p]+s*sizes[p];len[j*6+p]=sizes[p];}
        }
        cold_taken=taken;
        if(cold_overlap) {
            std::vector<int> group(size_t(jobs),0);
            void *ticket=cold_io.submit(cold_store,jobs,files.data(),offsets.data(),dst.data(),len.data(),6,group.data(),1);
            if(!ticket){release_taken();throw std::runtime_error("Reading cold experts from NVMe failed");}
            return ticket;
        }
        if(cold_io.read(cold_store,jobs,files.data(),offsets.data(),dst.data(),len.data(),6)) {
            release_taken();
            throw std::runtime_error("Reading cold experts from NVMe failed");
        }
        land(tokens,t0);
        return nullptr;
    }
    // The reads ensure_resident started (overlap) end: their experts are in RAM. Counted: the time the forward waited.
    void finish_reads(void *ticket,int tokens) {
        const auto t0=std::chrono::steady_clock::now();
        if(cold_io.finish(cold_store,ticket)){release_taken();throw std::runtime_error("Reading cold experts from NVMe failed");}
        land(tokens,t0);
    }
    void release_taken() {   // the reserved slots are free again; the experts they held are still on NVMe
        std::lock_guard<std::mutex> g(resident_lock);
        for(int s:cold_taken)expert_of[s]=-1;
    }
    void land(int tokens,std::chrono::steady_clock::time_point t0) {
        {
            std::lock_guard<std::mutex> g(resident_lock);
            for(size_t q=0;q<cold_miss.size();q++){slot_of[cold_miss[q]]=cold_taken[q];expert_of[cold_taken[q]]=cold_miss[q];}
        }
        const double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        cold_loads+=(long long)cold_miss.size();cold_seconds+=spent;
        if(tokens<=8){cold_small_loads+=(long long)cold_miss.size();cold_small_seconds+=spent;}
    }
    int attach_cold(void *store,const ColdIO *io,const int32_t *files,const long long *offsets,const float *scores,
                    float half_life) override {
        std::unique_lock<std::mutex> guard(call_lock,std::try_to_lock);
        if(!guard.owns_lock()||!store||!io||!io->read||!io->submit||!io->finish||!files||!offsets||!(half_life>0))return -1;
        // per expert and node part ([e*nodes+n]): where the part's six pieces are in the store
        cold_file.assign(files,files+size_t(experts)*nodes);cold_offset.assign(offsets,offsets+size_t(experts)*nodes);
        for(int e=0;e<experts;e++)if(slot_of[e]<0&&!stored(e))return -1;   // in neither RAM nor the store
        cold_score.assign(experts,0.0);
        if(scores)for(int e=0;e<experts;e++)cold_score[e]=scores[e];
        cold_stamp.assign(experts,0);cold_mark.assign(experts,0);pins.assign(slots,0);
        cold_log=std::log(0.5)/double(half_life);
        cold_io=*io;cold_store=store;
        const char *overlap=std::getenv("XTIER_COLD_OVERLAP");
        cold_overlap=!overlap||std::string(overlap)!="0";
        return 0;
    }
    // A slot another thread reads (the GPU pool's copy of an expert): its expert stays until unpin. -1: not in RAM.
    int pin(int expert) override {
        std::lock_guard<std::mutex> g(resident_lock);
        if(expert<0||expert>=experts)return -1;
        const int s=slot_of[expert];
        if(s>=0&&!pins.empty())pins[s]++;
        return s;
    }
    void unpin(int slot) override {
        std::lock_guard<std::mutex> g(resident_lock);
        if(slot>=0&&slot<int(pins.size())&&pins[slot]>0)pins[slot]--;
    }
    // forwards, experts read from NVMe, seconds spent reading; the same for the forwards of at most 8 tokens
    void cold_stats(double *out) override {
        out[0]=double(cold_forwards);out[1]=double(cold_loads);out[2]=cold_seconds;
        out[3]=double(cold_small_forwards);out[4]=double(cold_small_loads);out[5]=cold_small_seconds;
    }
    // Placement, from the plan's one caller between forwards (the call lock guards it): expert into slot; the expert
    // the slot held, -1 if none; -2 invalid, -3 a forward in progress.
    int place(int expert,int slot) override {
        if(expert<0||expert>=experts||slot<0||slot>=slots)return -2;
        std::unique_lock<std::mutex> guard(call_lock,std::try_to_lock);
        if(!guard.owns_lock())return -3;
        std::lock_guard<std::mutex> g(resident_lock);
        if(!pins.empty()&&pins[slot])return -3;
        const int previous=expert_of[slot];
        if(previous>=0)slot_of[previous]=-1;
        if(slot_of[expert]>=0)expert_of[slot_of[expert]]=-1;
        slot_of[expert]=slot;expert_of[slot]=expert;
        return previous;
    }
    int slot_table(int32_t *table) override {
        std::unique_lock<std::mutex> guard(call_lock,std::try_to_lock);
        if(!guard.owns_lock())return -3;
        std::copy(slot_of.begin(),slot_of.end(),table);return slots;
    }
    // One slot's six pieces on one node (gate, up, gate scales, up scales, down, down scales), as in xtier_shard.
    int slot_pieces(int slot,int node,void **pointers,size_t *bytes) const override {
        if(slot<0||slot>=slots||node<0||node>=nodes)return -1;
        const size_t H=hidden,I=inter,IN=I/nodes,HN=H/nodes,S=size_t(slot);const auto &s=shards[node];
        uint8_t *base[6]={s.gate.get(),s.up.get(),s.sg.get(),s.su.get(),s.down.get(),s.sd.get()};
        const size_t sizes[6]={IN*(H/2),IN*(H/2),IN*(H>>scale_shift),IN*(H>>scale_shift),HN*(I/2),HN*(I>>scale_shift)};
        for(int j=0;j<6;j++){pointers[j]=base[j]+S*sizes[j];bytes[j]=sizes[j];}
        return 0;
    }
    void shard(int node,void **pointers,size_t *bytes) const {
        const size_t E=slots,H=hidden,I=inter,IN=I/nodes,HN=H/nodes;const auto &s=shards[node];
        const Bytes *all[6]={&s.gate,&s.up,&s.sg,&s.su,&s.down,&s.sd};
        size_t sizes[6]={E*IN*(H/2),E*IN*(H/2),E*IN*(H>>scale_shift),E*IN*(H>>scale_shift),E*HN*(I/2),E*HN*(I>>scale_shift)};
        for(int j=0;j<6;j++){pointers[j]=all[j]->get();bytes[j]=sizes[j];}
    }
};
thread_local std::string last_error;
struct Handle {std::shared_ptr<PlanBase> plan;};
}

// Per-ISA builds (XTIER_IMPL_PREFIX=avx512_ / avx2_, one library each) export prefixed names that
// src/dispatch.cpp looks up after dlopen; a single build without the prefix exports xtier_* itself.
#ifdef XTIER_IMPL_PREFIX
#define XTIER_CAT2(a,b) a##b
#define XTIER_CAT(a,b) XTIER_CAT2(a,b)
#define XTIER_API(name) XTIER_CAT(XTIER_IMPL_PREFIX,name)
#pragma GCC visibility push(default)
#else
#define XTIER_API(name) name
#endif
extern "C" {
const char *XTIER_API(xtier_last_error)(){return last_error.c_str();}
const char *XTIER_API(xtier_isa)(){return XTIER_AVX512?"avx512":"avx2";}
void *XTIER_API(xtier_create3)(int hidden,int inter,int experts,int threads,float swiglu_limit,int scale_format,
                    const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                    const float *g13,const float *g2) {
    try {
#if XTIER_AVX512
        if(!__builtin_cpu_supports("avx512bf16")||!__builtin_cpu_supports("avx512bw"))
            throw std::runtime_error("AVX512 BF16/BW required");
#else
        if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma"))
            throw std::runtime_error("AVX2/FMA required");
#endif
        if(experts<10||experts>MAX_EXPERTS||threads<1||threads>128||!w13||!w2||!s13||!s2||!g13||!g2)
            throw std::runtime_error("Invalid create arguments");
        if(scale_format==1)return new Handle{std::make_shared<Plan<true>>(hidden,inter,experts,threads,swiglu_limit,w13,w2,s13,s2,g13,g2,1)};
        if(scale_format!=0)throw std::runtime_error("Unknown scale format");
        return new Handle{std::make_shared<Plan<false>>(hidden,inter,experts,threads,swiglu_limit,w13,w2,s13,s2,g13,g2,0)};
    }catch(const std::exception &e){last_error=e.what();return nullptr;}
}
void *XTIER_API(xtier_create4)(int hidden,int inter,int experts,int slots,int threads,float swiglu_limit,int scale_format,
                    const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                    const float *g13,const float *g2,const int32_t *resident) {
    try {
#if XTIER_AVX512
        if(!__builtin_cpu_supports("avx512bf16")||!__builtin_cpu_supports("avx512bw"))
            throw std::runtime_error("AVX512 BF16/BW required");
#else
        if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma"))
            throw std::runtime_error("AVX2/FMA required");
#endif
        if(experts<10||experts>MAX_EXPERTS||slots<1||slots>experts||threads<1||threads>128||!w13||!w2||!s13||!s2||!g13||!g2
           ||!resident)
            throw std::runtime_error("Invalid create arguments");
        if(scale_format==1)
            return new Handle{std::make_shared<Plan<true>>(hidden,inter,experts,threads,swiglu_limit,w13,w2,s13,s2,g13,g2,1,
                                                           slots,resident)};
        if(scale_format!=0)throw std::runtime_error("Unknown scale format");
        return new Handle{std::make_shared<Plan<false>>(hidden,inter,experts,threads,swiglu_limit,w13,w2,s13,s2,g13,g2,0,
                                                        slots,resident)};
    }catch(const std::exception &e){last_error=e.what();return nullptr;}
}
int XTIER_API(xtier_cold_attach)(void *plan,void *store,void *io,const int32_t *files,const long long *offsets,
                                  const float *scores,float half_life) {
    if(!plan){last_error="Null plan";return -1;}
    int rc=static_cast<Handle*>(plan)->plan->attach_cold(store,static_cast<const ColdIO*>(io),files,offsets,scores,half_life);
    if(rc)last_error="Invalid cold attach (an expert in neither RAM nor the store, or a forward in progress)";
    return rc;
}
int XTIER_API(xtier_slot_pin)(void *plan,int expert) {return plan?static_cast<Handle*>(plan)->plan->pin(expert):-1;}
void XTIER_API(xtier_slot_unpin)(void *plan,int slot) {if(plan)static_cast<Handle*>(plan)->plan->unpin(slot);}
void XTIER_API(xtier_cold_plan_stats)(void *plan,double *out) {if(plan&&out)static_cast<Handle*>(plan)->plan->cold_stats(out);}
int XTIER_API(xtier_place)(void *plan,int expert,int slot) {
    if(!plan){last_error="Null plan";return -2;}
    int rc=static_cast<Handle*>(plan)->plan->place(expert,slot);
    if(rc<-1)last_error=rc==-3?"Placement during a forward":"Invalid placement";
    return rc;
}
int XTIER_API(xtier_slot_table)(void *plan,int32_t *table) {
    if(!plan||!table){last_error="Invalid slot table query";return -1;}
    int rc=static_cast<Handle*>(plan)->plan->slot_table(table);
    if(rc<0)last_error="Slot table query during a forward";
    return rc;
}
int XTIER_API(xtier_slot_pieces)(void *plan,int slot,int node,void **pointers,size_t *bytes) {
    if(!plan||!pointers||!bytes){last_error="Invalid slot query";return -1;}
    int rc=static_cast<Handle*>(plan)->plan->slot_pieces(slot,node,pointers,bytes);
    if(rc<0)last_error="Invalid slot or node";
    return rc;
}
void *XTIER_API(xtier_create2)(int hidden,int inter,int experts,int threads,float limit,
                    const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                    const float *g13,const float *g2) {
    return XTIER_API(xtier_create3)(hidden,inter,experts,threads,limit,0,w13,w2,s13,s2,g13,g2);
}
int XTIER_API(xtier_set_swiglu)(void *plan,float alpha,float beta) {
    if(!plan||!(alpha>0.f)){last_error="Invalid SwiGLU parameters";return -1;}
    static_cast<Handle*>(plan)->plan->set_swiglu(alpha,beta);return 0;
}
int XTIER_API(xtier_shape)(void *plan,int *out) {
    if(!plan||!out){last_error="Invalid shape query";return -1;}
    static_cast<Handle*>(plan)->plan->shape(out);return 0;
}
int XTIER_API(xtier_read_probe)(void *plan,int m,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
    try {
        if(!plan)throw std::runtime_error("Null plan");
        auto &p=*static_cast<Handle*>(plan)->plan;
        p.set_read_only(true);int rc=p.forward(m,k,x,ids,rw,y);p.set_read_only(false);return rc;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}
int XTIER_API(xtier_forward)(void *plan,int m,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
    try {if(!plan)throw std::runtime_error("Null plan");return static_cast<Handle*>(plan)->plan->forward(m,k,x,ids,rw,y);}
    catch(const std::exception &e){last_error=e.what();return -1;}
}
void *XTIER_API(xtier_retain)(void *plan) {
    try {if(!plan)throw std::runtime_error("Null plan");return new Handle{static_cast<Handle*>(plan)->plan};}
    catch(const std::exception&e){last_error=e.what();return nullptr;}
}
void XTIER_API(xtier_destroy)(void *plan){delete static_cast<Handle*>(plan);}
void XTIER_API(xtier_last_timings)(void *plan,double *out){if(plan&&out)static_cast<Handle*>(plan)->plan->timings(out);}
int XTIER_API(xtier_localize)(void *plan) {
    try {if(!plan)throw std::runtime_error("Null plan");return static_cast<Handle*>(plan)->plan->localize();}
    catch(const std::exception &e){last_error=e.what();return -1;}
}
int XTIER_API(xtier_shard)(void *plan,int node,void **pointers,size_t *bytes) {
    if(!plan||node<0||node>=static_cast<Handle*>(plan)->plan->node_count()||!pointers||!bytes){last_error="Invalid shard query";return -1;}
    static_cast<Handle*>(plan)->plan->shard(node,pointers,bytes);return 0;
}
int XTIER_API(xtier_nodes)(void *plan) {
    if(!plan){last_error="Null plan";return -1;}
    return static_cast<Handle*>(plan)->plan->node_count();
}
int XTIER_API(xtier_worker_cpus)(int threads,int *cpus,int capacity) {
    try {
        int cores=node_cores(threads);
        if(!cpus||capacity<threads)throw std::runtime_error("CPU buffer too small");
        for(int i=0;i<threads;i++)cpus[i]=worker_cpu(i,cores);
        return threads;
    }catch(const std::exception &e){last_error=e.what();return -1;}
}
}
#ifdef XTIER_IMPL_PREFIX
#pragma GCC visibility pop
#endif
