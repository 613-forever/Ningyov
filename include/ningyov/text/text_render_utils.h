// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file text_render_utils.h
/// @brief Font-related function isolated from the library @c FreeType.

#pragma once
#ifndef NINGYOV_TEXT_RENDER_UTILS_H
#define NINGYOV_TEXT_RENDER_UTILS_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

struct FT_FaceRec_; // exposed from Freetype.

namespace ningyov { namespace font {

/// @cond
struct FaceCloser {
  void operator()(FT_FaceRec_* face);
};
/// @endcond
/// @brief Auto releasing @c FT_Face.
using Face = std::unique_ptr<FT_FaceRec_, FaceCloser>;

/// @brief Initialize font library.
void init();

struct FontInfo {
  std::string filename;
  std::uint16_t pixelWidth, pixelHeight;
};

}

namespace config {

extern std::string FONT_DIRECTORY;
extern std::vector<font::FontInfo> FONT_NAMES_AND_SIZES;

} }

#endif //NINGYOV_TEXT_RENDER_UTILS_H
