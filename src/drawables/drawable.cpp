// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/drawable.h>

#include <dialog_video_generator/image.h>
#include <dialog_video_generator/math/pos_arith.h>
#include <utility>

namespace dialog_video_generator { namespace drawable {

Static::Static(Image sprite) : image(std::move(sprite)) {}

Static::~Static() = default;

EntireImage::EntireImage(const std::string& dir, const std::string& name, unsigned int mul, Vec2i offset) :
    EntireImage(Image{RawImage{}, mul, offset}) {
  image.raw.load(dir, name, false);
}

EntireImage::EntireImage(Image spr) : Static(std::move(spr)) {}

EntireImage::~EntireImage() = default;

void EntireImage::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
//  Range range = image.range(offset);
  image.addTask(offset, false, alpha, false, tasks);
}

Texture::Texture(const std::string& dir, const std::string& name, unsigned int mul, Vec2i offset) :
    Texture(Image{RawImage{}, mul, offset}) {
  image.raw.load(dir, name, false);
}

Texture::Texture(Image spr) : Static(std::move(spr)) {}

Texture::~Texture() = default;

void Texture::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  image.addTask(offset, true, alpha, false, tasks);
}

Translated::Translated(std::shared_ptr<Drawable> target, Vec2i offset)
: target(std::move(target)), translatedOffset(offset) {}

Translated::~Translated() = default;

void Translated::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  target->addTask(translatedOffset + offset, alpha, tasks);
}

} }
