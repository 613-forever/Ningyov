// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file cuda_utils.h
/// @brief Utils to use GPU to render images.

#pragma once
#ifndef NINGYOV_CUDA_UTILS_H
#define NINGYOV_CUDA_UTILS_H

#include <memory>
#include <common613/memory.h>
#include <ningyov/common.h>

namespace ningyov {

/// @brief Smart pointer to GPU memory. Only returned by @ref allocateMemory methods
using CudaMemory = std::shared_ptr<unsigned char>;

namespace cuda {

/// @brief Automatic closer for @ref CudaMemory.
struct Closer {
  void operator()(unsigned char* p);
};

/// @brief Allocate GPU memory.
CudaMemory allocateMemory(unsigned int size);

/// @brief Allocate GPU memory of @f$ 4hw @f$ bytes to accommodate a 4-channel \c uint8_t image.
inline CudaMemory allocateMemory(unsigned int h, unsigned int w) {
  return allocateMemory(h * w * 4);
}

/// @brief Allocate GPU memory of @f$ 3hw @f$ bytes to accommodate a 3-channel \c uint8_t image.
inline CudaMemory allocateMemory3(unsigned int h, unsigned int w) {
  return allocateMemory(h * w * 3);
}

/// @brief Allocate GPU memory of @p size bytes and copy contents from CPU memory pointed by @p p.
CudaMemory copyFromCPUMemory(const void* p, unsigned int size);

/// @overload
inline CudaMemory copyFromCPUMemory(const common613::Memory& memory) {
  return copyFromCPUMemory(memory.data(), memory.size());
}

/// @brief Initialize CUDA.
void init();

/// @brief check CUDA error.
void checkCudaError();

}

namespace config {
/// @brief How many batches a row will be split into.
/// Automatically evaluated using @ref WIDTH and @ref GPU_MAX_THREAD_PER_BLOCK.
extern std::uint16_t WIDTH_BATCHES;
/// @brief How many pixels are calculated at once, i.e. batch size.
/// Automatically evaluated using @ref WIDTH and @ref GPU_MAX_THREAD_PER_BLOCK.
extern std::uint16_t THREAD_PER_BLOCK;
}

#if 0
/// @brief Show the first pixel for a PNG in CPU memory.
/// @note DEBUG function. No-op in normal use for performance.
void showFirstPixelForLocal(const unsigned char* memory, const char* prefix);

/// @brief Show the first pixel for a PNG in GPU memory.
/// @note DEBUG function. No-op in normal use for performance.
void showFirstPixelForCuda(const unsigned char* cudaMemory, const char* prefix);
#else
inline void showFirstPixelForCuda(const unsigned char*, const char*) {}
inline void showFirstPixelForLocal(const unsigned char*, const char*) {}
#endif

/// @brief Wrapper for a layer task.
struct DrawTask {
  int y0, x0, ht, wt;
  unsigned int mul;
  unsigned int alpha;
  bool useAlphaChannel, useExtraAlpha;
  bool skip, flip, useOnlyOneColor, isLinearFilter;
  const unsigned char* image;
};

/// @brief Wrapper for a layer task.
struct TextTask {
  int y0, x0, ht, wt;
  std::uint8_t r, g, b;
  bool reuse;
  const unsigned char* image;
};

namespace cuda {

/// @brief Render tasks in @p pTask onto buffers in @p pDst, and copy RGB channels into @p rgb.
/// @note Cuda @c __HOST__ function.
void renderTasks(unsigned char** pDst, const DrawTask* pTask, std::size_t taskNum, unsigned char* rgb);

/// @brief Copy RGB channel from 4-channel GPU image into a 3-channel one.
/// @note Cuda @c __HOST__ function.
void copyRGBChannel(unsigned char* dst, const unsigned char* src);

/// @brief Merge glyph masks in @p pGlyph with the mask @p pSrc, into the mask buffer @p pDst.
/// @note Cuda @c __HOST__ function.
void mergeGlyphMasks(unsigned char** pDst, const TextTask* pTask, std::size_t taskNum,
                     unsigned int height, unsigned int width);

}

}

#endif //NINGYOV_CUDA_UTILS_H
