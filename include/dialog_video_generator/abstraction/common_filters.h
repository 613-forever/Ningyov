// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_COMMON_FILTERS_H
#define DIALOGVIDEOGENERATOR_COMMON_FILTERS_H

#include <dialog_video_generator/drawables/filters.h>

namespace dialog_video_generator { namespace abstraction {

class BlackAndWhiteFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  BlackAndWhiteFilter();
};

class NostalgicFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  NostalgicFilter();
};

class DuskFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  DuskFilter();
};

} }

#endif //DIALOGVIDEOGENERATOR_COMMON_FILTERS_H
