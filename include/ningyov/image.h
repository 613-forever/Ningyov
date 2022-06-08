// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Images, abstraction for images on GPU.

#pragma once
#ifndef NINGYOV_IMAGE_H
#define NINGYOV_IMAGE_H

#include <string>
#include <unordered_map>
#include <atomic>
#include <iosfwd>
#include <ningyov/cuda/cuda_utils.h>
#include <ningyov/math/pos_utils.h>

/// @cond
// Forward declarations for png-related, to avoid including png++ headers here.
namespace png {
typedef unsigned char png_byte;
typedef png_byte byte;
template< typename T >
struct basic_rgb_pixel;
typedef basic_rgb_pixel<byte> rgb_pixel;
template <class pixel>
class solid_pixel_buffer;
template <class pixel, class pixel_buffer_type>
class image;
}
/// @endcond

namespace ningyov { namespace image {

/**
 * @class RawImage3
 * @brief Images with RGB channels in GPU, with a CPU buffer.
 */
struct RawImage3 {
  /// @brief Type of PNG CPU buffer.
  using PNG = png::image<png::rgb_pixel, png::solid_pixel_buffer<png::rgb_pixel>>;

  /// @brief Image size.
  Size size{Size::of(0, 0)};
  /// @brief Image on GPU.
  CudaMemory memory{};

  /// @brief Async saves the image into @p filename under @p dir, decrement @p counter when finished.
  void write(const std::string& dir, const std::string& filename, std::atomic_int& counter);
  /// @brief Async saves the image into @p fullPathname, decrement @p counter when finished.
  void writeFullPathname(const std::string& fullPathname, std::atomic_int& counter);
  /// @brief Writes the image into a stream.
  void write(std::ostream& os);

  /// @brief Image on CPU, lazy initialized.
  std::shared_ptr<PNG> lazyCPUMemory{};
  /// @brief Whether the CPU buffer @ref lazyCPUMemory is up to update.
  bool validLazyCPUMemory{false};
  /// @brief Invalidates the CPU buffer @ref lazyCPUMemory.
  void invalidateLazyCPUMemory() {
    validLazyCPUMemory = false;
  }

  /// @brief Allocates an @ref RawImage3 object (Factory function).
  static std::shared_ptr<RawImage3> allocate(UDim height, UDim width) {
    return std::make_shared<RawImage3>(RawImage3{Size{{height, width}}, cuda::allocateMemory3(height, width)});
  }
private:
  PNG& getLazyCPUMemory();
};

/**
 * @class RawImage
 * @brief Images with RGBA channels in GPU.
 */
struct RawImage {
  /// @brief Image size.
  Size size{Size::of(0, 0)};
  /// @brief Image on GPU.
  CudaMemory memory{};

  /// @brief Loads an image from @p filename under @p dir.
  /// @param regis whether we register it in an image cache.
  void load(const std::string& dir, const std::string& filename, bool regis);

  /// @brief Async saves the image into @p filename under @p dir, decrement @p counter when finished.
  /// @note Alpha channel will be ignored, since we only save opaque frames and videos.
  [[deprecated]] void write(const std::string& dir, const std::string& filename, std::atomic_int& counter) {
    toRawImage3().write(dir, filename, counter);
  }

  /// @brief Converts and copies into a @ref RawImage3 object.
  RawImage3 toRawImage3();

  /// @brief Allocates an @ref RawImage object (Factory function).
  static std::shared_ptr<RawImage> allocate(UDim height, UDim width) {
    return std::make_shared<RawImage>(RawImage{Size::of(height, width), cuda::allocateMemory(height, width)});
  }
};

/// @brief Image cache.
extern std::unordered_map<std::string, RawImage> registeredImages;

/**
 * @class Image
 * @brief Abstraction of images in viewport, i.e., combinations of a @ref RawImage
 * with magnifier, position and flip information.
 */
struct Image {
  /// @brief The image itself.
  RawImage raw;
  /// @brief The multiplier when pasted.
  unsigned int mul{1};
  /// @brief The position to paste.
  Vec2i pos{Vec2i::of(0, 0)};
  /// @brief Whether the image is flipped horizontally when pasted.
  bool flipX{false};

  /**
   * @brief Append a task into @p tasks with external position transformations.
   * @param offset Extra offset when we paste it.
   * @param withAlpha,extraAlpha Extra alpha information (0-16) used when pasting.
   * @param flip Extra horizontal flipping.
   */
  void addTask(Vec2i offset, bool withAlpha, unsigned int extraAlpha, bool flip, std::vector<DrawTask>& tasks) const;
};

/**
 * @class ColorImage
 * @brief Abstraction of images, all whose pixels are of a single color, along with extra position information.
 * @note A generated counterpart to the loaded class @ref Image,
 * but multiplier and flip fields are removed because they make no sense.
 */
struct ColorImage {
  /// @brief Image size.
  Size size{Size::of(0, 0)};
  /// @brief Color on GPU, R-G-B-A, uint8 * 4.
  CudaMemory color{};
  /// @brief The position to paste.
  Vec2i pos{Vec2i::of(0, 0)};

  /**
   * @brief Append a task into @p tasks with external position transformations.
   * @param offset Extra offset when we paste it.
   * @param extraAlpha Extra alpha information (0-16) used when pasting.
   */
  void addTask(Vec2i offset, unsigned int extraAlpha, std::vector<DrawTask>& tasks) const;
};

}

using image::RawImage;
using image::Image;
using image::ColorImage;

}

#endif //NINGYOV_IMAGE_H
