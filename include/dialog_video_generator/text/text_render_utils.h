// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file text_render_utils.h
/// @brief Font-related function isolated from the library @c FreeType.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H
#define DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H

#include <string>

namespace dialog_video_generator { namespace font {

/// @brief Initialize font library.
void init(const std::string& fontDir,
          const std::string& fontNameForChinese,
          const std::string& fontNameForJapanese);

} }

#endif //DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H
