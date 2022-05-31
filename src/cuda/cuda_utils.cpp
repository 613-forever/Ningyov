// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/cuda/cuda_utils.h>

#include <cuda_runtime.h>
#include <dialog_video_generator/config.h>

namespace dialog_video_generator { namespace cuda {

void Closer::operator()(unsigned char* p) {
  BOOST_LOG_TRIVIAL(trace) << "Releasing CUDA memory @ " << (void*)(p);
  cudaFree(p);
}

CudaMemory allocateMemory(unsigned int size) {
  unsigned char* result = nullptr;
  if (auto error = cudaMalloc(&result, size)) {
    COMMON613_FATAL("CudaMalloc failed. Error code: {}.", error);
  }
  COMMON613_REQUIRE(result != nullptr, "CudaMalloc might have exceeded memory limit. size={}.", size);
  BOOST_LOG_TRIVIAL(trace) << "Allocated CUDA memory @ " << (void*)(result);
  return CudaMemory{result, cuda::Closer()};
}

CudaMemory copyFromCPUMemory(const void* memory, unsigned int size) {
  CudaMemory result = allocateMemory(size);
  cudaMemcpy(result.get(), memory, size, cudaMemcpyHostToDevice);
  return result;
}

void init() {
  using namespace config;

  if (WIDTH_BATCHES == 0) {
    WIDTH_BATCHES = (
        WIDTH <= GPU_MAX_THREAD_PER_BLOCK * 1 ? 1 :
        WIDTH <= GPU_MAX_THREAD_PER_BLOCK * 2 ? 2 :
        WIDTH <= GPU_MAX_THREAD_PER_BLOCK * 4 ? 4 :
        WIDTH <= GPU_MAX_THREAD_PER_BLOCK * 8 ? 8 :
        WIDTH <= GPU_MAX_THREAD_PER_BLOCK * 16 ? 16 :
        32
    );
  }
  THREAD_PER_BLOCK = WIDTH / WIDTH_BATCHES;

  BOOST_LOG_TRIVIAL(trace) << "Initializing Cuda...";
  int count;
  cudaGetDeviceCount(&count);
  COMMON613_REQUIRE(count > 0, "Cuda device not found.");
  int i;
  for (i = 0; i < count; i++) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 1) {
        if (prop.maxThreadsPerBlock < GPU_MAX_THREAD_PER_BLOCK) {
          BOOST_LOG_TRIVIAL(warning) << fmt::format(
                "Device [{}] allows only {} threads, which is not larger than required {} threads set in configure.",
                i, prop.maxThreadsPerBlock, GPU_MAX_THREAD_PER_BLOCK);
        } else {
          BOOST_LOG_TRIVIAL(info) << fmt::format(
                "Selected Device [{}], allowing {} threads per block, and {}x{}x{}-sized grids.",
                i, prop.maxThreadsPerBlock, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
          break;
        }
      }
    }
  }
  COMMON613_REQUIRE(i < count, "Not any available Cuda device.");
  if (auto error = cudaSetDevice(i)) {
    COMMON613_FATAL("CudaSetDevice failed. Error code: {}.", error);
  }
}

}

namespace config {
std::uint16_t WIDTH_BATCHES = 0;
std::uint16_t THREAD_PER_BLOCK = 0;
}

#if 0
void showFirstPixelForCuda(const unsigned char* cudaMemory, const char* prefix) {
  unsigned char buffer[4];
  cudaMemcpy(buffer, cudaMemory, 4, cudaMemcpyDeviceToHost);
  BOOST_LOG_TRIVIAL(trace) << fmt::format("{} {:p}[{},{},{};{}]", prefix, cudaMemory,
                                          buffer[0], buffer[1], buffer[2], buffer[3]);
}

void showFirstPixelForLocal(const unsigned char* memory, const char* prefix) {
  BOOST_LOG_TRIVIAL(trace) << fmt::format("{} {:p}[{},{},{};{}]", prefix, memory,
                                          memory[0], memory[1], memory[2], memory[3]);
}
#else
void showFirstPixelForCuda(const unsigned char*, const char*) {}
void showFirstPixelForLocal(const unsigned char*, const char*) {}
#endif


}
