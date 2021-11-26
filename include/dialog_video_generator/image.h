// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_IMAGE_H
#define DIALOGVIDEOGENERATOR_IMAGE_H

#include <string>
#include <unordered_map>
#include <atomic>
#include <iosfwd>
#include <dialog_video_generator/cuda/cuda_utils.h>
#include <dialog_video_generator/math/pos_utils.h>
#include <png.hpp>
#include <solid_pixel_buffer.hpp>

namespace dialog_video_generator { namespace image {

struct RawImage3 {
  using PNG = png::image<png::rgb_pixel, png::solid_pixel_buffer<png::rgb_pixel>>;
  Size size{std::array<UDim, 2>{0, 0}};
  CudaMemory memory{};
  std::shared_ptr<PNG> lazyCPUMemory{};
  void write(const std::string& dir, const std::string& filename, std::atomic_int& counter);
  void writeFullPathname(const std::string& fullPathname, std::atomic_int& counter);
  void write(std::ostream& os);

  static std::shared_ptr<RawImage3> allocate(UDim height, UDim width) {
    return std::make_shared<RawImage3>(RawImage3{Size{{height, width}}, cuda::allocateMemory3(height, width)});
  }
private:
  PNG& getLazyCPUMemory();
};

struct RawImage {
  Size size{std::array<UDim, 2>{0, 0}};
  CudaMemory memory{};

//  void copyTo(const CudaMemory& target) const;
  void load(const std::string& dir, const std::string& filename, bool regis);
  [[deprecated]] void write(const std::string& dir, const std::string& filename, std::atomic_int& counter);
  RawImage3 toRawImage3();
  [[deprecated]] void copyToMemory3(const std::function<void(CudaMemory&&, std::size_t)>& callback);

  static std::shared_ptr<RawImage> allocate(UDim height, UDim width) {
    return std::make_shared<RawImage>(RawImage{Size{height, width}, cuda::allocateMemory(height, width)});
  }
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
