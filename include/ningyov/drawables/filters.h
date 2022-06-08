// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Utils for filter drawables.

#pragma once
#ifndef NINGYOV_FILTERS_H
#define NINGYOV_FILTERS_H

#include <ningyov/drawable.h>
#include <ningyov/cuda/cuda_utils.h>

namespace ningyov { namespace drawable {

/**
 * @brief Base class for filters.
 *
 * Filters are objects that transform colors for images, which are specially handled in rendering.
 */
class Filter : public Static {
public:
  ~Filter() override;
};

/**
 * @brief Base class for filters which can be represented in a matrix multiplication with an additive bias.
 *
 * The transformation should be able to be written as follows:
 * \f[ \begin{bmatrix} r' \\ g' \\ b' \end{bmatrix} = \begin{bmatrix}
 * c_{rr} & c_{rg} & c_{rb} & b_{r+} \\
 * c_{gr} & c_{gg} & c_{gb} & b_{g+} \\
 * c_{br} & c_{bg} & c_{bb} & b_{b+} \\
 * \end{bmatrix} \begin{bmatrix} r \\ g \\ b \\ 1 \end{bmatrix} \f]
 * where coefficients \f$ c_{\bullet\bullet} \f$ are in [-2, 2), and biases \f$ b_{\bullet} \f$ in [-128, 128).
 * The renderer would truncate output \f$ r',g',b' \f$ to [0, 255] automatically.
 *
 * @note The matrix should always be a constant matrix in row-order, with coefficients in 64x, and biases in 1x.
 */
class LinearFilter : public Filter {
public:
  explicit LinearFilter(const std::int8_t (& data)[12]);
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
  ~LinearFilter() override;
private:
  CudaMemory coefficients; // (64x) rr, gr, br; rg, gg, bg; rb, gb, bb; (1x) r+, g+ b+.
};

} }

#endif //NINGYOV_FILTERS_H
