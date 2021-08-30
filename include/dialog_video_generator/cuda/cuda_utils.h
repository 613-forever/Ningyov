// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_CUDA_UTILS_H
#define DIALOGVIDEOGENERATOR_CUDA_UTILS_H

#include <memory>
#include <common613/memory.h>
#include <dialog_video_generator/common.h>

namespace dialog_video_generator {

using CudaMemory = std::shared_ptr<unsigned char>;

namespace cuda {

struct Closer {
  void operator()(unsigned char* p);
};

CudaMemory allocateMemory(unsigned int size);

inline CudaMemory allocateMemory(unsigned int h, unsigned int w) {
  return allocateMemory(h * w * 4);
}

inline CudaMemory allocateMemory3(unsigned int h, unsigned int w) {
  return allocateMemory(h * w * 3);
}

CudaMemory copyFromCPUMemory(const void* p, unsigned int size);

inline CudaMemory copyFromCPUMemory(const common613::Memory& memory) {
  return copyFromCPUMemory(memory.data(), memory.size());
}

void init();

}

namespace config {
extern std::uint16_t WIDTH_BATCHES;
extern std::uint16_t THREAD_PER_BLOCK;
}

void showFirstPixelForLocal(const unsigned char* memory, const char* prefix);

void showFirstPixelForCuda(const unsigned char* cudaMemory, const char* prefix);

struct DrawTask {
  int y0, x0, ht, wt;
  unsigned int mul;
  unsigned int alpha;
  bool useAlphaChannel, useAsAlphaOnly, useAsAlphaAndUseColorText, useExtraAlpha;
  bool skip, flip;
  const unsigned char* image;
};

namespace cuda {

void renderTasks(unsigned char** pDst, const DrawTask* pTask, size_t taskNum);

void copyRGBChannel(unsigned char* dst, const unsigned char* src);

}

}

#endif //DIALOGVIDEOGENERATOR_CUDA_UTILS_H
