// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_IMAGE_H
#define DIALOGVIDEOGENERATOR_IMAGE_H

#include <string>
#include <unordered_map>
#include <atomic>
#include <dialog_video_generator/cuda/cuda_utils.h>
#include <dialog_video_generator/math/pos_utils.h>

namespace dialog_video_generator { namespace image {

struct RawImage {
  Size size{std::array<UDim, 2>{0, 0}};
  CudaMemory memory{};

  void copyTo(const CudaMemory& target) const;
  void load(const std::string& dir, const std::string& filename, bool regis);
  void write(const std::string& dir, const std::string& filename, std::atomic_int& counter);
//  void addToWriteQueue();
  void copyToMemory3(const std::function<void(CudaMemory&&, std::size_t)>& callback);
};

extern std::unordered_map<std::string, RawImage> regisImages;

struct Image {
  RawImage raw;
  unsigned int mul{1};
  Vec2i pos{std::array<Dim, 2>{0, 0}};
  bool flipX{false};

  void addTask(Vec2i offset, bool withAlpha, unsigned int extraAlpha, bool flip, std::vector<DrawTask>& tasks) const;
  Range range(Vec2i offset) const;
};

}

using image::RawImage;
using image::Image;

}

#endif // !DIALOGVIDEOGENERATOR_DRAWABLE_H
