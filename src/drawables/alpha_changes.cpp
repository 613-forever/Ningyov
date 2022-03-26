// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/drawables/alpha_changes.h>

namespace dialog_video_generator { namespace drawable {

FadeIn::FadeIn(const std::shared_ptr<Drawable>& target, const Frames& duration)
: AlphaChange(target, duration) {}

FadeIn::~FadeIn() = default;

int FadeIn::calculateAlpha(Frames timeInShot) const {
  return static_cast<int>((timeInShot.x() * 16 + dur.x() - 1) / dur.x());
}

FadeOut::FadeOut(const std::shared_ptr<Drawable>& target, const Frames& duration)
: AlphaChange(target, duration) {}

FadeOut::~FadeOut() = default;

int FadeOut::calculateAlpha(Frames timeInShot) const {
  return static_cast<int>(((dur.x() - timeInShot.x()) * 16 + dur.x() - 1) / dur.x());
}

} }
