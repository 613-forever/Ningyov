// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

/// @file
/// @brief Animations with alpha changing, derivation classes of @ref AlphaChange .

#pragma once
#ifndef DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H
#define DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace drawable {

/**
 * @brief Linear fade-in animation.
 */
class FadeIn : public AlphaChange {
public:
  FadeIn(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeIn() override;
  int calculateAlpha(Frames timeInScene) const override;
};

/**
 * @brief Linear fade-out animation.
 */
class FadeOut : public AlphaChange {
public:
  FadeOut(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeOut() override;
  int calculateAlpha(Frames timeInScene) const override;
};

} }

#endif //DIALOGVIDEOGENERATOR_ALPHA_CHANGES_H
