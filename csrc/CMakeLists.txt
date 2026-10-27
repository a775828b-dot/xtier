cmake_minimum_required(VERSION 3.18)
project(xtier_engine LANGUAGES CXX)
find_package(Threads REQUIRED)

set(XTIER_PREFILL_MICRO_TOKENS "8" CACHE STRING "CPU expert microbatch size: 4 or 8")
if(NOT XTIER_PREFILL_MICRO_TOKENS MATCHES "^(4|8)$")
  message(FATAL_ERROR "XTIER_PREFILL_MICRO_TOKENS must be 4 or 8")
endif()
# engine/engine.cpp is built once per ISA into its own library (libxtier_engine_avx512.so, libxtier_engine_avx2.so);
# libxtier_engine.so (engine/dispatch.cpp, baseline x86-64) exports the API and dlopens the build picked at run time
# from its own directory. Separate libraries matter: inline functions both builds use (e.g. std::to_string) would
# otherwise be merged into one copy compiled for one of the ISAs. Keep the flags in step with the CPU checks in
# engine/dispatch.cpp.
set(XTIER_AVX512_FLAGS "-march=x86-64-v4;-mavx512bf16;-mavx512vbmi;-mtune=znver4" CACHE STRING "AVX512 kernel build flags")
set(XTIER_AVX2_FLAGS "-march=x86-64-v3;-mtune=znver2" CACHE STRING "AVX2 kernel build flags")
# Down rows per microkernel iteration in the AVX512 build: 1 (one-row, the release build since lkqmoe 0.5.5) or 4.
set(XTIER_AVX512_DOWN_R4 "1" CACHE STRING "Down microkernel rows of the AVX512 build: 1, 2 or 4")
foreach(isa avx512 avx2)
  string(TOUPPER ${isa} ISA)
  add_library(xtier_engine_${isa} SHARED engine/engine.cpp)
  target_include_directories(xtier_engine_${isa} PRIVATE include)
  target_compile_definitions(xtier_engine_${isa} PRIVATE XTIER_PREFILL_MICRO_TOKENS=${XTIER_PREFILL_MICRO_TOKENS} XTIER_IMPL_PREFIX=${isa}_)
  if(isa STREQUAL "avx512")
    target_compile_definitions(xtier_engine_${isa} PRIVATE XTIER_DOWN_R4=${XTIER_AVX512_DOWN_R4})
  endif()
  target_compile_features(xtier_engine_${isa} PRIVATE cxx_std_17)
  target_compile_options(xtier_engine_${isa} PRIVATE -O3 ${XTIER_${ISA}_FLAGS} -fvisibility=hidden -Wall -Wextra)
  # bind every reference inside the library to its own definitions (its own std inline copies)
  target_link_options(xtier_engine_${isa} PRIVATE -Wl,-Bsymbolic)
  target_link_libraries(xtier_engine_${isa} PRIVATE Threads::Threads)
endforeach()
# cold/store.cpp: cold experts on NVMe (O_DIRECT reads into RAM slots), no ISA-specific code.
add_library(xtier_engine SHARED engine/dispatch.cpp cold/store.cpp)
add_dependencies(xtier_engine xtier_engine_avx512 xtier_engine_avx2)
target_include_directories(xtier_engine PRIVATE include)
target_compile_features(xtier_engine PRIVATE cxx_std_17)
target_compile_options(xtier_engine PRIVATE -O2 -Wall -Wextra)
target_link_libraries(xtier_engine PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

# CPU-GPU mailbox (mapped host memory + stream memory operations, CUDA-graph capturable) and the expert pool
# command ring. Needs the CUDA runtime headers / library and the driver library.
set(XTIER_CUDA_ROOT "" CACHE PATH "CUDA runtime directory containing include/ and lib/")
if(XTIER_CUDA_ROOT)
  # pip's nvidia/cu13 has only libcudart.so.13 (no libcudart.so link), the driver may lack the libcuda.so dev link
  find_library(XTIER_CUDART NAMES cudart libcudart.so.13 libcudart.so.12 PATHS "${XTIER_CUDA_ROOT}/lib" "${XTIER_CUDA_ROOT}/lib64"
               NO_DEFAULT_PATH REQUIRED)
  find_library(XTIER_CUDA_DRIVER NAMES cuda libcuda.so.1 REQUIRED)
  add_library(xtier_mailbox SHARED mailbox/mailbox.cpp)
  target_include_directories(xtier_mailbox PRIVATE include "${XTIER_CUDA_ROOT}/include")
  target_compile_features(xtier_mailbox PRIVATE cxx_std_17)
  target_compile_options(xtier_mailbox PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(xtier_mailbox PRIVATE xtier_engine ${XTIER_CUDART} ${XTIER_CUDA_DRIVER} Threads::Threads)
  set_target_properties(xtier_mailbox PROPERTIES BUILD_RPATH "$ORIGIN;${XTIER_CUDA_ROOT}/lib")
endif()

# Row tables on NVMe (xtier.tables: DeepSeek V4.1 Engram, Qwen3.8 PLE n-grams): exact rows read with O_DIRECT, a
# bounded row cache; the lookup calls no CUDA API (a cudaLaunchHostFunc node).
add_library(xtier_tables SHARED tables/rows.cpp)
target_compile_features(xtier_tables PRIVATE cxx_std_17)
target_compile_options(xtier_tables PRIVATE -O2 -fvisibility=hidden -Wall -Wextra)
target_link_libraries(xtier_tables PRIVATE Threads::Threads)
