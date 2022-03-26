// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/common_filters.h>

namespace dialog_video_generator { namespace abstraction {

BlackAndWhiteFilter::BlackAndWhiteFilter() : LinearFilter(filterData) {}

std::int8_t BlackAndWhiteFilter::filterData[12] = {
    14, 46, 5, 0,
    14, 46, 5, 0,
    14, 46, 5, 0,
};

NostalgicFilter::NostalgicFilter() : LinearFilter(filterData) {}

std::int8_t NostalgicFilter::filterData[12] = {
    25, 49, 12, 0,
    22, 44, 11, 0,
    17, 34, 8, 0,
};

DuskFilter::DuskFilter() : LinearFilter(filterData) {}

std::int8_t DuskFilter::filterData[12] = {
    70, 0, 0, 0,
    -6, 65, -2, 0,
    3, 9, 48, 0,
};

} }
