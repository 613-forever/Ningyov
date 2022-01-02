// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_FILTERS_H
#define DIALOGVIDEOGENERATOR_FILTERS_H

#include <dialog_video_generator/drawable.h>
#include <dialog_video_generator/cuda/cuda_utils.h>

namespace dialog_video_generator { namespace drawable {

class Filter : public Static {
public:
  ~Filter() override;
};

class LinearFilter : public Filter {
public:
  explicit LinearFilter(const std::int8_t (& data)[12]);
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
  ~LinearFilter() override;
private:
  CudaMemory coefficients; // (64x) rr, gr, br; rg, gg, bg; rb, gb, bb; (1x) r+, g+ b+.
};

} }

#endif //DIALOGVIDEOGENERATOR_FILTERS_H
