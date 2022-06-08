// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/cuda/cuda_utils.h>
#include <ningyov/config.h>

namespace ningyov { namespace cuda {

namespace {

__device__ void copyPixelRGB(unsigned char* dst, const unsigned char* bg, unsigned int WIDTH,
                             unsigned int y, unsigned int x) {
  unsigned int index = y * WIDTH + x;
  dst[index * 3 + 0] = bg[index * 4 + 0];
  dst[index * 3 + 1] = bg[index * 4 + 1];
  dst[index * 3 + 2] = bg[index * 4 + 2];
}

__device__ void renderPixel(unsigned char* dst, const unsigned char* bg, const DrawTask* task, unsigned int WIDTH,
                            unsigned int y, unsigned int x) {
  unsigned int index = y * WIDTH + x;
  if (task->isLinearFilter) {
    std::int16_t r = (
        std::int16_t(std::int8_t(task->image[0])) * bg[index * 4] +
            std::int16_t(std::int8_t(task->image[1])) * bg[index * 4 + 1] +
            std::int16_t(std::int8_t(task->image[2])) * bg[index * 4 + 2]) / 64 + std::int8_t(task->image[3]);
    dst[index * 4 + 0] = r < 0 ? 0 : r > 255 ? 255 : std::uint8_t(r);
    std::int16_t g = (
        std::int16_t(std::int8_t(task->image[4])) * bg[index * 4] +
            std::int16_t(std::int8_t(task->image[5])) * bg[index * 4 + 1] +
            std::int16_t(std::int8_t(task->image[6])) * bg[index * 4 + 2]) / 64 + std::int8_t(task->image[7]);
    dst[index * 4 + 1] = g < 0 ? 0 : g > 255 ? 255 : std::uint8_t(g);
    std::int16_t b = (
        std::int16_t(std::int8_t(task->image[8])) * bg[index * 4] +
            std::int16_t(std::int8_t(task->image[9])) * bg[index * 4 + 1] +
            std::int16_t(std::int8_t(task->image[10])) * bg[index * 4 + 2]) / 64 + std::int8_t(task->image[11]);
    dst[index * 4 + 2] = b < 0 ? 0 : b > 255 ? 255 : std::uint8_t(b);
  } else {
    int deltaY = static_cast<int>(y) - task->y0, deltaX = static_cast<int>(x) - task->x0;
    if (deltaX < 0 || deltaX >= task->wt * task->mul || deltaY < 0 || deltaY >= task->ht * task->mul) {
      dst[index * 4 + 0] = bg[index * 4 + 0];
      dst[index * 4 + 1] = bg[index * 4 + 1];
      dst[index * 4 + 2] = bg[index * 4 + 2];
    } else {
      int deltaYInTexture = deltaY / int(task->mul), deltaXInTexture = deltaX / int(task->mul);
      unsigned textureIndex = task->wt * deltaYInTexture + (task->flip ? task->wt - 1 - deltaXInTexture : deltaXInTexture);
      textureIndex *= !task->useOnlyOneColor;
      if (task->useAlphaChannel) {
        if (task->useExtraAlpha) {
          std::uint16_t alpha = task->image[textureIndex * 4 + 3] * task->alpha / 16;
          dst[index * 4 + 0] = (task->image[textureIndex * 4 + 0] * alpha + bg[index * 4 + 0] * (255 - alpha)) / 255;
          dst[index * 4 + 1] = (task->image[textureIndex * 4 + 1] * alpha + bg[index * 4 + 1] * (255 - alpha)) / 255;
          dst[index * 4 + 2] = (task->image[textureIndex * 4 + 2] * alpha + bg[index * 4 + 2] * (255 - alpha)) / 255;
        } else {
          std::uint16_t alpha = task->image[textureIndex * 4 + 3];
          dst[index * 4 + 0] = (task->image[textureIndex * 4 + 0] * alpha + bg[index * 4 + 0] * (255 - alpha)) / 255;
          dst[index * 4 + 1] = (task->image[textureIndex * 4 + 1] * alpha + bg[index * 4 + 1] * (255 - alpha)) / 255;
          dst[index * 4 + 2] = (task->image[textureIndex * 4 + 2] * alpha + bg[index * 4 + 2] * (255 - alpha)) / 255;
        }
      } else {
        if (task->useExtraAlpha) {
          std::uint16_t alpha = task->alpha;
          dst[index * 4 + 0] = (task->image[textureIndex * 4 + 0] * alpha + bg[index * 4 + 0] * (16 - alpha)) / 16;
          dst[index * 4 + 1] = (task->image[textureIndex * 4 + 1] * alpha + bg[index * 4 + 0] * (16 - alpha)) / 16;
          dst[index * 4 + 2] = (task->image[textureIndex * 4 + 2] * alpha + bg[index * 4 + 0] * (16 - alpha)) / 16;
        } else {
          dst[index * 4 + 0] = task->image[textureIndex * 4 + 0];
          dst[index * 4 + 1] = task->image[textureIndex * 4 + 1];
          dst[index * 4 + 2] = task->image[textureIndex * 4 + 2];
        }
      }
    }
  }
}

__global__ void renderKernel(unsigned char** pDst, const DrawTask* pTask, size_t taskNum, unsigned char* rgb,
                             unsigned int THREAD_PER_BLOCK, unsigned int WIDTH) {
  unsigned int x = threadIdx.x + THREAD_PER_BLOCK * blockIdx.y, y = blockIdx.x;
  __shared__ unsigned char* bg;
  bg = *pDst;
  for (int i = 0; i < taskNum; ++i, ++pTask) {
    ++pDst;
    if (!pTask->skip) {
      renderPixel(*pDst, bg, pTask, WIDTH, y, x);
      __syncthreads();
      bg = *pDst;
    }
  }
  if (rgb) {
    copyPixelRGB(rgb, bg, WIDTH, y, x);
  }
}

} // anonymous namespace

__host__ void renderTasks(unsigned char** pDst, const DrawTask* pTask, size_t taskNum, unsigned char* rgb) {
  dim3 blockInGrid(config::HEIGHT, config::WIDTH_BATCHES);
  dim3 threadInBlock(config::THREAD_PER_BLOCK);
  renderKernel<<<blockInGrid, threadInBlock>>>(pDst, pTask, taskNum, rgb, config::THREAD_PER_BLOCK, config::WIDTH);
}

namespace {

__global__ void copyRGBKernel(unsigned char* dst, const unsigned char* src,
                              unsigned int THREAD_PER_BLOCK, unsigned int WIDTH) {
  unsigned int x = threadIdx.x + THREAD_PER_BLOCK * blockIdx.y, y = blockIdx.x;
  copyPixelRGB(dst, src, WIDTH, y, x);
}

} // anonymous namespace

__host__ void copyRGBChannel(unsigned char* dst, const unsigned char* src) {
  dim3 blockInGrid(config::HEIGHT, config::WIDTH_BATCHES);
  dim3 threadInBlock(config::THREAD_PER_BLOCK);
  copyRGBKernel<<<blockInGrid, threadInBlock>>>(dst, src, config::THREAD_PER_BLOCK, config::WIDTH);
}

namespace {

__device__ void zeroOutMask(unsigned char* dst, unsigned int width, unsigned int y, unsigned int x) {
  unsigned int index = y * width + x;
  dst[index * 4 + 0] = 0;
  dst[index * 4 + 1] = 0;
  dst[index * 4 + 2] = 0;
  dst[index * 4 + 3] = 0;
}

__device__ void mergeMask(unsigned char* dst, const unsigned char* bg, const TextTask* task,
                          unsigned int width, unsigned int y, unsigned int x) {
  unsigned int index = y * width + x;
  int deltaY = static_cast<int>(y) - task->y0, deltaX = static_cast<int>(x) - task->x0;
  if (deltaX < 0 || deltaX >= task->wt || deltaY < 0 || deltaY >= task->ht) {
    dst[index * 4 + 3] = bg[index * 4 + 3];
    dst[index * 4 + 0] = bg[index * 4 + 0];
    dst[index * 4 + 1] = bg[index * 4 + 1];
    dst[index * 4 + 2] = bg[index * 4 + 2];
  } else {
    // clip only, do not use alpha blending
    unsigned int deltaYInTexture = deltaY, deltaXInTexture = deltaX; // no scaling
    unsigned textureIndex = task->wt * deltaYInTexture + deltaXInTexture;
    dst[index * 4 + 3] = task->image[textureIndex];
    dst[index * 4 + 0] = task->r;
    dst[index * 4 + 1] = task->g;
    dst[index * 4 + 2] = task->b;
  }
}

__global__ void mergeMaskKernel(unsigned char** pDst, const TextTask* pTask, std::size_t taskNum,
                                unsigned int width, unsigned int threadPerBlock) {
  unsigned int x = threadIdx.x + threadPerBlock * blockIdx.y, y = blockIdx.x;
  __shared__ unsigned char* bg;
  bg = *pDst;
  zeroOutMask(*pDst, width, y, x);
  for (int i = 0; i < taskNum; ++i, ++pTask) {
    if (!pTask->reuse) {
      ++pDst;
    }
    mergeMask(*pDst, bg, pTask, width, y, x);
    __syncthreads();
    bg = *pDst;
  }
}

} // anonymous namespace

__host__ void mergeGlyphMasks(unsigned char** pDst, const TextTask* pTask, size_t taskNum,
                              unsigned int height, unsigned int width) {
  unsigned int batchCount =
      width < config::THREAD_PER_BLOCK ? 1 :
      width < 2 * config::THREAD_PER_BLOCK ? 2 :
      width < 4 * config::THREAD_PER_BLOCK ? 4 :
      width < 8 * config::THREAD_PER_BLOCK ? 8 : 16;
  assert(width % batchCount == 0); // will be difficult to code if it does not divide.
  unsigned int threadPerBlock = width / batchCount;
  dim3 blockInGrid(height, batchCount);
  dim3 threadInBlock(threadPerBlock);
  mergeMaskKernel<<<blockInGrid, threadInBlock>>>(pDst, pTask, taskNum, width, threadPerBlock);
}

} }
