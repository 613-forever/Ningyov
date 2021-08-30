// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/text/text_render_details.h>

namespace dialog_video_generator { namespace font {

Library library{};
Face faceForChineseText{};
Face faceForJapaneseText{};

void init(const std::string& fontDir,
          const std::string& fontNameForChinese,
          const std::string& fontNameForJapanese) {
  BOOST_LOG_TRIVIAL(trace) << "Initializing FreeType...";
  FT_Library pLibrary;
  if (auto error = FT_Init_FreeType(&pLibrary)) {
    COMMON613_FATAL("FreeType Init failed. Error code: {}.", error);
  }
  library.reset(pLibrary);

  faceForChineseText = openFace(fontDir + fontNameForChinese);
//  FT_Set_Char_Size(faceForChineseText.get(), 0, 54 * 64, 72, 72);
  FT_Set_Pixel_Sizes(faceForChineseText.get(), 66, 72);
  faceForJapaneseText = openFace(fontDir + fontNameForJapanese);
//  FT_Set_Char_Size(faceForJapaneseText.get(), 0, 54 * 64, 72, 72);
  FT_Set_Pixel_Sizes(faceForJapaneseText.get(), 66, 72);
}

Face openFace(const std::string& filePathname) {
  BOOST_LOG_TRIVIAL(trace) << "Loading a typeface \"" << filePathname << "\".";
  FT_Face face;
  FT_Error error = FT_New_Face(library.get(), filePathname.c_str(), 0, &face);
  if (error == FT_Err_Unknown_File_Format) {
    COMMON613_FATAL("Unsupported file format reported by FreeType.");
  } else if (error) {
    COMMON613_FATAL("Erroneous font file reported by FreeType. Error code: {}.", error);
  }
  BOOST_LOG_TRIVIAL(trace) << "Loaded typeface is allocated at " << (void*)face;
  return Face(face);
}

FT_GlyphSlot loadGlyph(const Face& face, char32_t codePoint) {
  auto glyphIndex = FT_Get_Char_Index(face.get(), codePoint);
  FT_Load_Glyph(face.get(), glyphIndex, FT_LOAD_DEFAULT);
  if (face->glyph->format != FT_GLYPH_FORMAT_BITMAP) {
    FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
  }
  assert(face->glyph->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY);
  return face->glyph;
}

} }
