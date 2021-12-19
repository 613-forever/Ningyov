// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H
#define DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace drawable {

class FadeIn : public AlphaChange {
public:
  FadeIn(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeIn() override;
  int calculateAlpha(Frames timeInScene) const;
};

class FadeOut : public AlphaChange {
public:
  FadeOut(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeOut() override;
  int calculateAlpha(Frames timeInScene) const;
};

} }

#endif //DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H
