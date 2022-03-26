// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file action_animations.h
/// @brief Implementations for animations in actions.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H
#define DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace abstraction {

/**
 * @brief Animation in shouting actions.
 *
 * Shouting is an action jumping up and fall back to the ground.
 */
class Shouting : public drawable::Movement {
public:
  explicit Shouting(std::shared_ptr<Drawable> target);
  ~Shouting() override;
  Vec2i calculateOffset(Frames timeInShot) const override;
private:
  constexpr static const long double LENGTH_SECOND = 0.5;
};

/**
 * @brief Shortcut to make a Shouting object.
 * @related Shouting
 */
inline std::shared_ptr<Shouting> animateShouting(std::shared_ptr<Drawable> target) {
  return std::make_shared<Shouting>(std::move(target));
}

/**
 * @brief Animation in murmuring actions.
 *
 * Murmuring is an action trembling.
 */
class Murmuring : public drawable::Movement {
public:
  explicit Murmuring(std::shared_ptr<Drawable> target);
  ~Murmuring() override;
  Vec2i calculateOffset(Frames timeInShot) const override;
private:
  constexpr static const long double LENGTH_SECOND = 2;
};

/**
 * @brief Shortcut to make a Murmuring object.
 * @related Murmuring
 */
inline std::shared_ptr<Murmuring> animateMurmuring(std::shared_ptr<Drawable> target) {
  return std::make_shared<Murmuring>(std::move(target));
}

} }

#endif //DIALOGVIDEOGENERATOR_ACTION_ANIMATIONS_H
