// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Animations with alpha changing, derivation classes of @ref AlphaChange .

#pragma once
#ifndef NINGYOV_ALPHA_CHANGES_H
#define NINGYOV_ALPHA_CHANGES_H

#include <ningyov/drawable.h>

namespace ningyov { namespace drawable {

/**
 * @brief Linear fade-in animation.
 */
class FadeIn : public AlphaChange {
public:
  FadeIn(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeIn() override;
  int calculateAlpha(Frames timeInShot) const override;
};

/**
 * @brief Linear fade-out animation.
 */
class FadeOut : public AlphaChange {
public:
  FadeOut(const std::shared_ptr<Drawable>& target, const Frames& duration);
  ~FadeOut() override;
  int calculateAlpha(Frames timeInShot) const override;
};

} }

#endif //NINGYOV_ALPHA_CHANGES_H
