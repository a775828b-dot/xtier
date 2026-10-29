// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments
// renamed only, so the kernels compile to the same code.
// One libxtier.so for hosts with and without AVX512: src/xtier.cpp is built twice, into
// libxtier_avx512.so (AVX512 BF16/VBMI) and libxtier_avx2.so (AVX2/FMA), with prefixed API names
// (avx512_ / avx2_). This file, compiled for baseline x86-64, exports the public xtier_* API and
// forwards every call to the build chosen on first use, loaded with dlopen from this library's own
// directory: AVX512 when the CPU has every extension that build was compiled for, else AVX2.
// XTIER_ISA=avx512|avx2 forces one; an unsupported choice makes every create fail with the reason
// in xtier_last_error. Only the chosen library is ever loaded, so no code of the other runs.
#include "xtier_engine.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
struct Impl {
    const char *name=nullptr;
    const char *(*last_error)();
    const char *(*isa)();
    void *(*create3)(int,int,int,int,float,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,const float*);
    void *(*create2)(int,int,int,int,float,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,const float*);
    void *(*create4)(int,int,int,int,int,float,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,
                     const float*,const int32_t*);
    int (*place)(void*,int,int);
    int (*slot_table)(void*,int32_t*);
    int (*slot_pieces)(void*,int,int,void**,size_t*);
    int (*cold_attach)(void*,void*,void*,const int32_t*,const long long*,const float*,float);
    int (*slot_pin)(void*,int);
    void (*slot_unpin)(void*,int);
    void (*cold_plan_stats)(void*,double*);
    int (*set_swiglu)(void*,float,float);
    int (*shape)(void*,int*);
    int (*read_probe)(void*,int,int,const uint16_t*,const int32_t*,const float*,float*);
    int (*forward)(void*,int,int,const uint16_t*,const int32_t*,const float*,float*);
    void *(*retain)(void*);
    void (*destroy)(void*);
    void (*last_timings)(void*,double*);
    int (*localize)(void*);
    int (*shard)(void*,int,void**,size_t*);
    int (*nodes)(void*);
    int (*worker_cpus)(int,int*,int);
};
Impl impl_storage;
std::string error;

// Must match the compile flags of the two builds (CMakeLists.txt XTIER_AVX512_FLAGS /
// XTIER_AVX2_FLAGS). Feature by feature: GCC 11 has no x86-64-v3 / -v4 names here (the levels' F16C,
// LZCNT and MOVBE come with every AVX2 + FMA + BMI2 CPU).
bool has_avx2() {
    return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")&&__builtin_cpu_supports("bmi")&&
           __builtin_cpu_supports("bmi2");
}
bool has_avx512() {
    return has_avx2()&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&
           __builtin_cpu_supports("avx512cd")&&__builtin_cpu_supports("avx512dq")&&
           __builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512bf16")&&
           __builtin_cpu_supports("avx512vbmi");
}

std::string own_directory() {
    Dl_info info{};
    if(!dladdr(reinterpret_cast<void*>(&has_avx2),&info)||!info.dli_fname)return ".";
    std::string path=info.dli_fname;
    size_t slash=path.rfind('/');
    return slash==std::string::npos?".":path.substr(0,slash);
}

template<class F> bool bind(void *handle,const std::string &prefix,const char *name,F &slot) {
    void *symbol=dlsym(handle,(prefix+name).c_str());
    if(!symbol){error="xtier: "+prefix+name+" missing in the kernel library";return false;}
    slot=reinterpret_cast<F>(symbol);
    return true;
}

const Impl *load(const char *isa) {
    std::string path=own_directory()+"/libxtier_engine_"+isa+".so";
    void *h=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!h){const char *e=dlerror();error="xtier: cannot load "+path+": "+(e?e:"?");return nullptr;}
    Impl &m=impl_storage;std::string p=std::string(isa)+"_";
    bool ok=bind(h,p,"xtier_last_error",m.last_error)&&bind(h,p,"xtier_isa",m.isa)&&
            bind(h,p,"xtier_create3",m.create3)&&bind(h,p,"xtier_create2",m.create2)&&
            bind(h,p,"xtier_set_swiglu",m.set_swiglu)&&bind(h,p,"xtier_shape",m.shape)&&bind(h,p,"xtier_read_probe",m.read_probe)&&
            bind(h,p,"xtier_forward",m.forward)&&bind(h,p,"xtier_retain",m.retain)&&
            bind(h,p,"xtier_destroy",m.destroy)&&bind(h,p,"xtier_last_timings",m.last_timings)&&
            bind(h,p,"xtier_localize",m.localize)&&bind(h,p,"xtier_shard",m.shard)&&
            bind(h,p,"xtier_nodes",m.nodes)&&bind(h,p,"xtier_worker_cpus",m.worker_cpus)&&
            bind(h,p,"xtier_create4",m.create4)&&bind(h,p,"xtier_place",m.place)&&
            bind(h,p,"xtier_slot_table",m.slot_table)&&bind(h,p,"xtier_slot_pieces",m.slot_pieces)&&
            bind(h,p,"xtier_cold_attach",m.cold_attach)&&bind(h,p,"xtier_slot_pin",m.slot_pin)&&
            bind(h,p,"xtier_slot_unpin",m.slot_unpin)&&bind(h,p,"xtier_cold_plan_stats",m.cold_plan_stats);
    if(!ok)return nullptr;
    m.name=isa;
    return &m;
}

const Impl *choose() {
    __builtin_cpu_init();
    const char *env=std::getenv("XTIER_ISA");
    std::string want=env?env:"";
    const char *isa=nullptr;
    if(want==""||want=="auto") {
        isa=has_avx512()?"avx512":has_avx2()?"avx2":nullptr;
        if(!isa)error="xtier needs AVX2/FMA (x86-64-v3) or AVX512 BF16/VBMI";
    } else if(want=="avx512") {
        if(has_avx512())isa="avx512";
        else error="XTIER_ISA=avx512 but this CPU lacks AVX512 F/BW/CD/DQ/VL, BF16 or VBMI";
    } else if(want=="avx2") {
        if(has_avx2())isa="avx2";
        else error="XTIER_ISA=avx2 but this CPU lacks AVX2/FMA (x86-64-v3)";
    } else error="XTIER_ISA must be auto, avx512 or avx2";
    const Impl *chosen=isa?load(isa):nullptr;
    if(chosen)std::fprintf(stderr,"[xtier] CPU kernels: %s%s\n",chosen->name,want==""||want=="auto"?"":" (XTIER_ISA)");
    else std::fprintf(stderr,"[xtier] %s\n",error.c_str());
    return chosen;
}
const Impl *impl() {static const Impl *chosen=choose();return chosen;}
}

extern "C" {
const char *xtier_last_error(){auto p=impl();return p?p->last_error():error.c_str();}
const char *xtier_isa(){auto p=impl();return p?p->isa():"none";}
void *xtier_create3(int h,int i,int e,int t,float l,int f,
                     const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                     const float *g13,const float *g2) {
    auto p=impl();return p?p->create3(h,i,e,t,l,f,w13,w2,s13,s2,g13,g2):nullptr;
}
void *xtier_create2(int hidden,int inter,int experts,int threads,float swiglu_limit,
                     const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                     const float *g13,const float *g2) {
    auto p=impl();return p?p->create2(hidden,inter,experts,threads,swiglu_limit,w13,w2,s13,s2,g13,g2):nullptr;
}
void *xtier_create4(int h,int i,int e,int slots,int t,float l,int f,const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,
                    const uint8_t *s2,const float *g13,const float *g2,const int32_t *resident) {
    auto p=impl();return p?p->create4(h,i,e,slots,t,l,f,w13,w2,s13,s2,g13,g2,resident):nullptr;
}
// The store's reads for the engine build (engine.cpp's ColdIO: the same three pointers in this order).
struct ColdIO {
    int (*read)(void*,int,const int*,const long long*,void*const*,const size_t*,int);
    void *(*submit)(void*,int,const int*,const long long*,void*const*,const size_t*,int,const int*,int);
    int (*finish)(void*,void*);
};
int xtier_cold_attach(void *plan,void *store,const int32_t *files,const long long *offsets,const float *scores,float half_life) {
    static const ColdIO io{&xtier_cold_read,&xtier_cold_submit,&xtier_cold_finish};
    auto p=impl();
    return p?p->cold_attach(plan,store,const_cast<ColdIO*>(&io),files,offsets,scores,half_life):-1;
}
int xtier_slot_pin(void *plan,int expert){auto p=impl();return p?p->slot_pin(plan,expert):-1;}
void xtier_slot_unpin(void *plan,int slot){if(auto p=impl())p->slot_unpin(plan,slot);}
void xtier_cold_plan_stats(void *plan,double *out){if(auto p=impl())p->cold_plan_stats(plan,out);}
int xtier_place(void *plan,int expert,int slot){auto p=impl();return p?p->place(plan,expert,slot):-2;}
int xtier_slot_table(void *plan,int32_t *table){auto p=impl();return p?p->slot_table(plan,table):-1;}
int xtier_slot_pieces(void *plan,int slot,int node,void **pointers,size_t *bytes) {
    auto p=impl();return p?p->slot_pieces(plan,slot,node,pointers,bytes):-1;
}
int xtier_set_swiglu(void *plan,float alpha,float beta){auto p=impl();return p?p->set_swiglu(plan,alpha,beta):-1;}
int xtier_shape(void *plan,int *out){auto p=impl();return p?p->shape(plan,out):-1;}
int xtier_read_probe(void *plan,int m,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
    auto p=impl();return p?p->read_probe(plan,m,k,x,ids,rw,y):-1;
}
int xtier_forward(void *plan,int m,int k,const uint16_t *x,const int32_t *ids,const float *rw,float *y) {
    auto p=impl();return p?p->forward(plan,m,k,x,ids,rw,y):-1;
}
void *xtier_retain(void *plan){auto p=impl();return p?p->retain(plan):nullptr;}
void xtier_destroy(void *plan){if(auto p=impl())p->destroy(plan);}
void xtier_last_timings(void *plan,double *out){if(auto p=impl())p->last_timings(plan,out);}
int xtier_localize(void *plan){auto p=impl();return p?p->localize(plan):-1;}
int xtier_shard(void *plan,int node,void **pointers,size_t *bytes){auto p=impl();return p?p->shard(plan,node,pointers,bytes):-1;}
int xtier_nodes(void *plan){auto p=impl();return p?p->nodes(plan):-1;}
int xtier_worker_cpus(int threads,int *cpus,int capacity){auto p=impl();return p?p->worker_cpus(threads,cpus,capacity):-1;}
}
