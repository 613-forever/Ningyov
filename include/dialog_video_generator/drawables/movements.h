// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_MOVEMENTS_H
#define DIALOGVIDEOGENERATOR_MOVEMENTS_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace drawable {

class SimpleMovement : public Movement {
public:
  SimpleMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  SimpleMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~SimpleMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

class CubicEaseInMovement : public Movement {
public:
  CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~CubicEaseInMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

class CubicEaseOutMovement : public Movement {
public:
  CubicEaseOutMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames duration);
  CubicEaseOutMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames duration);
  ~CubicEaseOutMovement() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  Vec2i start, end;
};

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
