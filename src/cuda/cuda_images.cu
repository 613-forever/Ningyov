// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/cuda/cuda_utils.h>

namespace dialog_video_generator { namespace cuda {

namespace {

__device__ void copyPixelRGB(unsigned char* dst, const unsigned char* bg, unsigned int WIDTH, unsigned int y, unsigned int x) {
  unsigned int index = y * WIDTH + x;
  dst[index * 3 + 0] = bg[index * 4 + 0];
  dst[index * 3 + 1] = bg[index * 4 + 1];
  dst[index * 3 + 2] = bg[index * 4 + 2];
}

__device__ void renderPixel(unsigned char* dst, const unsigned char* bg, const DrawTask* task, unsigned int WIDTH, unsigned int y, unsigned int x) {
  unsigned int index = y * WIDTH + x;
  int deltaY = static_cast<int>(y) - task->y0, deltaX = static_cast<int>(x) - task->x0;
  if (deltaX < 0 || deltaX >= task->wt * task->mul || deltaY < 0 || deltaY >= task->ht * task->mul) {
    dst[index * 4 + 0] = bg[index * 4 + 0];
    dst[index * 4 + 1] = bg[index * 4 + 1];
    dst[index * 4 + 2] = bg[index * 4 + 2];
  } else {
    int deltaYInTexture = deltaY / int(task->mul), deltaXInTexture = deltaX / int(task->mul);
    unsigned int textureIndex = task->wt * deltaYInTexture + (task->flip ? task->wt - 1 - deltaXInTexture : deltaXInTexture);
    if (task->useAsAlphaOnly) {
      std::uint16_t alpha = task->image[textureIndex];
      if (task->useAsAlphaAndUseColorText) {
        dst[index * 4 + 0] = (255 - alpha) * bg[index * 4 + 0] / 255;
        dst[index * 4 + 1] = (255 - alpha) * bg[index * 4 + 1] / 255;
        dst[index * 4 + 2] = (255 - alpha) * bg[index * 4 + 2] / 255 + alpha;
      } else {
        dst[index * 4 + 0] = (255 - alpha) * bg[index * 4 + 0] / 255;
        dst[index * 4 + 1] = (255 - alpha) * bg[index * 4 + 1] / 255;
        dst[index * 4 + 2] = (255 - alpha) * bg[index * 4 + 2] / 255;
      }
    } else {
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

} }
