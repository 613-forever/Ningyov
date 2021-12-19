// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/drawables/alpha_changes.h>

namespace dialog_video_generator { namespace drawable {

FadeIn::FadeIn(const std::shared_ptr<Drawable>& target, const Frames& duration)
: AlphaChange(target, duration) {}

FadeIn::~FadeIn() = default;

int FadeIn::calculateAlpha(Frames timeInScene) const {
  return (timeInScene.x() * 16 + dur.x() - 1) / dur.x();
}

FadeOut::FadeOut(const std::shared_ptr<Drawable>& target, const Frames& duration)
: AlphaChange(target, duration) {}

FadeOut::~FadeOut() = default;

int FadeOut::calculateAlpha(Frames timeInScene) const {
  return ((dur.x() - timeInScene.x()) * 16 + dur.x() - 1) / dur.x();
}

} }
