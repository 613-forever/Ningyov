// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

/// @file
/// @brief Animations with position changing, derivation classes of @ref Movement .

#pragma once
#ifndef DIALOGVIDEOGENERATOR_MOVEMENTS_H
#define DIALOGVIDEOGENERATOR_MOVEMENTS_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace drawable {

/**
 * @brief Simple movement.
 *
 * Offset change is linear to time.
 */
class SimpleMovement : public Movement {
public:
  SimpleMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  SimpleMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~SimpleMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

/**
 * @brief Cubic ease-in movement.
 *
 * Offset change is cubic to time, like \f$ x^3 / t^3 \f$ .
 * It is a simplification for normal \f$ a (x-b)^3 / t^3 + c \f$ without parameter support for now.
 */
class CubicEaseInMovement : public Movement {
public:
  CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~CubicEaseInMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

/**
 * @brief Cubic ease-out movement.
 *
 * Offset change is cubic to time, like \f$ 1 - (t-x)^3 / t^3 \f$ .
 * It is a simplification for normal \f$ 1 - a (t-x-b)^3 / t^3 + c \f$ without parameter support for now.
 */
class CubicEaseOutMovement : public Movement {
public:
  CubicEaseOutMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  CubicEaseOutMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~CubicEaseOutMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

/**
 * @brief Cubic ease-in-out movement.
 *
 * Both before and after the center point, the offset change is cubic to time, like in ease-in and ease-out respective.
 * It is a simplification for normal ease-in-out without parameter support for now.
 */
class CubicEaseInOutMovement : public Movement {
public:
  CubicEaseInOutMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  CubicEaseInOutMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~CubicEaseInOutMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

} }

#endif //DIALOGVIDEOGENERATOR_MOVEMENTS_H
