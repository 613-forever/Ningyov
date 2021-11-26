// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/image.h>

#include <memory>
#include <thread>
#include <cuda_runtime.h>
#include <common613/compat/file_system.h>
#include <dialog_video_generator/math/pos_arith.h>

namespace dialog_video_generator { namespace image {

std::unordered_map<std::string, RawImage> regisImages(0x100); // NOLINT(cert-err58-cpp)

//void RawImage::copyTo(const CudaMemory& target) const {
//  cudaMemcpy(target.get(), memory.get(), size.h() * size.w() * Color4b::size, cudaMemcpyDeviceToDevice);
//}

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

void RawImage3::write(const std::string& dir, const std::string& filename, std::atomic_int& counter) {
  std::string fullPathname = dir + filename + ".png";
  writeFullPathname(fullPathname, counter);
}

void RawImage3::writeFullPathname(const std::string& fullPathname, std::atomic_int& counter) {
  BOOST_LOG_TRIVIAL(debug) << fmt::format("Trying writing rawImage3 \"{}\" from {}x{}@{:p}",
                                          fullPathname, size.h(), size.w(), memory.get());
  getLazyCPUMemory().write(fullPathname);
  BOOST_LOG_TRIVIAL(info) << fmt::format("Written: \"{}\".", fullPathname);
}

void RawImage3::write(std::ostream& os) {
  os.write(reinterpret_cast<const char*>(getLazyCPUMemory().get_pixbuf().get_bytes().data()), size.total() * 3);
}

RawImage3::PNG& RawImage3::getLazyCPUMemory() {
  if (!lazyCPUMemory) {
    UDim width = size.w(), height = size.h();
    lazyCPUMemory = std::make_shared<PNG>(width, height);
    auto& image = *lazyCPUMemory;
    COMMON613_REQUIRE(size.total() * 3 == image.get_pixbuf().get_bytes().size(), "Found mismatching size.");
    BOOST_LOG_TRIVIAL(trace) << fmt::format("Data in PNG object: [{},{},{}]", image.get_pixel(0, 0).red,
                                            image.get_pixel(0, 0).green, image.get_pixel(0, 0).blue);
    cudaMemcpy(const_cast<png::byte*>(image.get_pixbuf().get_bytes().data()),
               memory.get(), size.total() * 3, cudaMemcpyDeviceToHost);
  }
  return *lazyCPUMemory;
}

void RawImage::write(const std::string& dir, const std::string& filename, std::atomic_int& counter) {
  toRawImage3().write(dir, filename, counter);
}

RawImage3 RawImage::toRawImage3() {
  auto memory3 = cuda::allocateMemory3(size.h(), size.w());
  cuda::copyRGBChannel(memory3.get(), memory.get());
  return RawImage3{size, memory3, {}};
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
      false, (this->flipX != flip),
      raw.memory.get(),
  });
}

Range Image::range(Vec2i offset) const {
  return makeRange(pos + offset, raw.size);
}

} }
