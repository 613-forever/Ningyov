// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H
#define DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include <common613/assert.h>
#include <dialog_video_generator/text/text_render_details.h>

namespace dialog_video_generator { namespace font {

struct LibraryCloser {
  void operator()(FT_Library lib) {
    FT_Done_FreeType(lib);
  }
};
using Library = std::unique_ptr<std::remove_pointer_t<FT_Library>, LibraryCloser>;
extern Library library;

struct GlyphCloser {
  void operator()(FT_Glyph glyph) {
//    BOOST_LOG_TRIVIAL(trace) << "Releasing a glyph";
    FT_Done_Glyph(glyph);
  }
};
using Glyph = std::unique_ptr<std::remove_pointer_t<FT_Glyph>, GlyphCloser>;

struct FaceCloser {
  void operator()(FT_Face face) {
//    BOOST_LOG_TRIVIAL(trace) << "Releasing a typeface @" << (void*)face;
    FT_Done_Face(face);
  }
};
using Face = std::unique_ptr<std::remove_pointer_t<FT_Face>, FaceCloser>;
Face openFace(const std::string& filePathname);
extern Face faceForChineseText;
extern Face faceForJapaneseText;

FT_GlyphSlot loadGlyph(const Face& face, char32_t codePoint);

} }

#endif //DIALOGVIDEOGENERATOR_TEXT_RENDER_DETAILS_H
