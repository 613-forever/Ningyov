// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H
#define DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace abstraction {

class Shouting : public drawable::Movement {
public:
  explicit Shouting(std::shared_ptr<Drawable> target);
  ~Shouting() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  constexpr static const long double LENGTH_SECOND = 0.5;
};

inline std::shared_ptr<Shouting> animateShouting(std::shared_ptr<Drawable> target) {
  return std::make_shared<Shouting>(std::move(target));
}

class Murmuring : public drawable::Movement {
public:
  explicit Murmuring(std::shared_ptr<Drawable> target);
  ~Murmuring() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  constexpr static const long double LENGTH_SECOND = 2;
};

inline std::shared_ptr<Murmuring> animateMurmuring(std::shared_ptr<Drawable> target) {
  return std::make_shared<Murmuring>(std::move(target));
}

} }

#endif //DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H
