// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/drawable.h>

#include <ningyov/math/pos_arith.h>
#include <utility>

namespace ningyov { namespace drawable {

Static::~Static() = default;

StaticImage::StaticImage(Image sprite) : image(std::move(sprite)) {}

StaticImage::~StaticImage() = default;

Size StaticImage::staticSize() const {
  return image.raw.size;
}

EntireImage::EntireImage(const std::string& dir, const std::string& name, unsigned int mul, Vec2i offset) :
    EntireImage(Image{RawImage{}, mul, offset}) {
  image.raw.load(dir, name, false);
}

EntireImage::EntireImage(Image spr) : StaticImage(std::move(spr)) {}

EntireImage::~EntireImage() = default;

void EntireImage::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
//  Range range = image.range(offset);
  image.addTask(offset, false, alpha, false, tasks);
}

Texture::Texture(const std::string& dir, const std::string& name, unsigned int mul, Vec2i offset) :
    Texture(Image{RawImage{}, mul, offset}) {
  image.raw.load(dir, name, false);
}

Texture::Texture(Image spr) : StaticImage(std::move(spr)) {}

Texture::~Texture() = default;

void Texture::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  image.addTask(offset, true, alpha, false, tasks);
}

ColorRect::ColorRect(Color4b color) : colorImage{
  Size::of(config::HEIGHT, config::WIDTH),
  cuda::copyFromCPUMemory(&color, Color4b::COMMON613_INJECTED_SIZE),
  Vec2i{0, 0}
} {}

ColorRect::ColorRect(Color4b color, Vec2i offset, Size rect) : colorImage{
  rect,
  cuda::copyFromCPUMemory(&color, Color4b::COMMON613_INJECTED_SIZE),
  offset
} {}

ColorRect::~ColorRect() = default;

void ColorRect::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  colorImage.addTask(offset, alpha, tasks);
}

Translated::Translated(std::shared_ptr<Drawable> target, Vec2i offset)
: target(std::move(target)), translatedOffset(offset) {}

Translated::~Translated() = default;

void Translated::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  target->addTask(translatedOffset + offset, alpha, tasks);
}

} }
