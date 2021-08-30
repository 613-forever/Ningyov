// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/image.h>

#include <thread>
#include <cuda_runtime.h>
#include <png.hpp>
#include <solid_pixel_buffer.hpp>
#include <common613/compat/file_system.h>
#include <dialog_video_generator/math/pos_arith.h>

namespace dialog_video_generator { namespace image {

std::unordered_map<std::string, RawImage> regisImages(0x100);

void RawImage::copyTo(const CudaMemory& target) const {
  cudaMemcpy(target.get(), memory.get(), size.h() * size.w() * Color4b::size, cudaMemcpyDeviceToDevice);
}

void RawImage::load(const std::string& dir, const std::string& filename, bool regis) {
  {
    auto iter = regisImages.find(filename);
    if (iter != regisImages.end()) {
      *this = iter->second;
      return;
    }
  }
  std::string fullPathname = dir + filename + ".png";
  COMMON613_REQUIRE(boost::filesystem::exists(fullPathname), "Loading a non-existing file: {}.", fullPathname);
  BOOST_LOG_TRIVIAL(debug) << fmt::format("Loading image from \"{}\".", fullPathname);
  const png::image<png::rgba_pixel, png::solid_pixel_buffer<png::rgba_pixel>> image(fullPathname);
  auto pixelNumber = image.get_height() * image.get_width();
  if (size.total() != pixelNumber) {
    BOOST_LOG_TRIVIAL(debug) << fmt::format("Resizing image buffer from {} to {}.", size.total(), pixelNumber);
    memory = cuda::allocateMemory(image.get_height(), image.get_width());
  }
  size.h() = image.get_height();
  size.w() = image.get_width();

  showFirstPixelForLocal(image.get_pixbuf().get_bytes().data(), "Loaded data (CPU): ");
  cudaMemcpy(memory.get(), image.get_pixbuf().get_bytes().data(), pixelNumber * 4, cudaMemcpyHostToDevice);
  showFirstPixelForCuda(memory.get(), "Loaded data (GPU): ");
  if (regis) {
    regisImages[filename] = *this;
  }
  BOOST_LOG_TRIVIAL(trace) << fmt::format("Loaded an image from \"{}\".", fullPathname);
}

void RawImage::write(const std::string& dir, const std::string& filename, std::atomic_int& counter) {
  std::string fullPathname = dir + filename + ".png";
  UDim width = size.w(), height = size.h();
  BOOST_LOG_TRIVIAL(debug) << fmt::format("Trying writing \"{}\" from {}x{}@{:p}",
                                          fullPathname, height, width, memory.get());
//  assert(height == config::HEIGHT && width == config::WIDTH); // only to check if wrong image is written.

//  auto bufferSize = size.total() * 4;
//  Memory imageBuffer(bufferSize);
//  cudaMemcpy(imageBuffer.data(), memory.get(), bufferSize, cudaMemcpyDeviceToHost);

  auto memory3 = cuda::allocateMemory3(height, width);
  cuda::copyRGBChannel(memory3.get(), memory.get());
  size_t bufferSize = size.total() * 3;

  ++counter;
//  std::thread thr([fullPathname, &counter](Memory&& imageBuffer) {
//    png::image<png::rgb_pixel> image(WIDTH, HEIGHT);
//    for (png::uint_32 i = 0; i < HEIGHT; ++i) {
//      for (png::uint_32 j = 0; j < WIDTH; ++j) {
//        auto pixelIndex = i * WIDTH + j;
//        image.set_pixel(j, i, png::rgb_pixel(
//            png::byte(imageBuffer[pixelIndex * 4 + 0]),
//            png::byte(imageBuffer[pixelIndex * 4 + 1]),
//            png::byte(imageBuffer[pixelIndex * 4 + 2])
//        ));
//      }
//    }
//    BOOST_LOG_TRIVIAL(trace) << fmt::format("Data in PNG object: [{},{},{}]",
//                                            image.get_pixel(0, 0).red,
//                                            image.get_pixel(0, 0).green,
//                                            image.get_pixel(0, 0).blue);
//    image.write(fullPathname);
//    BOOST_LOG_TRIVIAL(info) << fmt::format("Written: \"{}\".", fullPathname);
//    --counter;
//  }, std::move(imageBuffer));
//  thr.detach();
  std::thread thr([fullPathname, &counter, bufferSize, width, height](CudaMemory&& memory3) {
    png::image<png::rgb_pixel, png::solid_pixel_buffer<png::rgb_pixel>> image(width, height);
//    png++ implementation-dependent cast. get_pixbuf().get_bytes() returns the very reference to the underlying vector.
    cudaMemcpy(const_cast<png::byte*>(image.get_pixbuf().get_bytes().data()),
               memory3.get(), bufferSize, cudaMemcpyDeviceToHost);
    BOOST_LOG_TRIVIAL(trace) << fmt::format("Data in PNG object: [{},{},{}]", image.get_pixel(0, 0).red,
                                            image.get_pixel(0, 0).green, image.get_pixel(0, 0).blue);
    image.write(fullPathname);
    BOOST_LOG_TRIVIAL(info) << fmt::format("Written: \"{}\".", fullPathname);
    --counter;
  }, std::move(memory3));
  thr.detach();
}

void RawImage::copyToMemory3(const std::function<void(CudaMemory&&, std::size_t)>& addToQueue) {
  auto memory3 = cuda::allocateMemory3(size.h(), size.w());
  cuda::copyRGBChannel(memory3.get(), memory.get());
  std::size_t bufferSize = size.total() * 3;
  addToQueue(std::move(memory3), bufferSize);
}

void Image::addTask(Vec2i offset, bool withAlpha, unsigned int extraAlpha, bool flip, std::vector<DrawTask>& tasks) const {
  assert(extraAlpha >= 0 && extraAlpha <= 16);
  tasks.emplace_back(DrawTask{
      pos.y() + offset.y(), pos.x() + offset.x(),
      raw.size.h(), raw.size.w(),
      mul,
      extraAlpha,
      withAlpha, false, false, extraAlpha < 16,
      false, (this->flipX == flip),
      raw.memory.get(),
  });
}

Range Image::range(Vec2i offset) const {
  return makeRange(pos + offset, raw.size);
}

} }
