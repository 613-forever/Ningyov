// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file text_render_details.h
/// @brief Wrapper for the library @c FreeType.
/// @note Use @ref text_render_utils.h if possible, to isolate @c FreeType from other codes.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H
#define DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include <common613/assert.h>
#include <dialog_video_generator/text/text_render_utils.h>

namespace dialog_video_generator { namespace font {

/// @cond
struct LibraryCloser {
  void operator()(FT_Library lib) {
    FT_Done_FreeType(lib);
  }
};
/// @endcond
/// @brief Auto releasing @c FT_Library.
using Library = std::unique_ptr<std::remove_pointer_t<FT_Library>, LibraryCloser>;
extern Library library;

/// @cond
struct GlyphCloser {
  void operator()(FT_Glyph glyph) {
//    BOOST_LOG_TRIVIAL(trace) << "Releasing a glyph";
    FT_Done_Glyph(glyph);
  }
};
/// @endcond
/// @brief Auto releasing @c FT_Glyph. Now unused. Maybe used when multiple fonts are involved.
using Glyph = std::unique_ptr<std::remove_pointer_t<FT_Glyph>, GlyphCloser>;

/// @cond
inline void FaceCloser::operator()(FT_FaceRec* face) {
  FT_Done_Face(face);
}
/// @endcond
/// @brief Opens a font face with specified @p filePathname.
Face openFace(const std::string& filePathname);

/// @brief Accesses @c FT_GlyphSlot directly.
FT_GlyphSlot loadGlyph(FT_Face face, char32_t codePoint);
FT_GlyphSlot loadGlyph(const Face& face, char32_t codePoint);
FT_GlyphSlot loadGlyph(std::size_t faceIndex, char32_t codePoint);

} }

#endif //DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H
