// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/text/text_render_details.h>

namespace ningyov { namespace config {
std::string FONT_DIRECTORY("./");
std::vector<font::FontInfo> FONT_NAMES_AND_SIZES{
    {"SourceHanSansJP-Regular.otf", 48, 48},
};
}

namespace font {

Library library{};
std::vector<Face> faces;

void init() {
  BOOST_LOG_TRIVIAL(trace) << "Initializing FreeType...";
  FT_Library pLibrary;
  if (auto error = FT_Init_FreeType(&pLibrary)) {
    COMMON613_FATAL("FreeType Init failed. Error code: {}.", error);
  }
  library.reset(pLibrary);

  for (auto& info : config::FONT_NAMES_AND_SIZES) {
    Face face = openFace(config::FONT_DIRECTORY + info.filename);
//    FT_Set_Char_Size(face.get(), 0, 54 * 64, 72, 72);
    FT_Set_Pixel_Sizes(face.get(), info.pixelWidth, info.pixelHeight);
    faces.emplace_back(std::move(face));
  }
}

Face openFace(const std::string& filePathname) {
  BOOST_LOG_TRIVIAL(trace) << "Loading a typeface \"" << filePathname << "\".";
  FT_Face face;
  FT_Error error = FT_New_Face(library.get(), filePathname.c_str(), 0, &face);
  if (error == FT_Err_Unknown_File_Format) {
    COMMON613_FATAL("Unsupported file format reported by Freetype: {}.", filePathname);
  } else if (error == FT_Err_Cannot_Open_Resource) {
    COMMON613_FATAL("Cannot open resource file by Freetype: {}.", filePathname);
  } if (error) {
    COMMON613_FATAL("Erroneous font file reported by Freetype: {}. Error code: {}.", filePathname, error);
  }
  BOOST_LOG_TRIVIAL(trace) << "Loaded typeface is allocated at " << (void*)face;
  return Face(face);
}

FT_GlyphSlot loadGlyph(FT_Face face, char32_t codePoint) {
  auto glyphIndex = FT_Get_Char_Index(face, codePoint);
  FT_Load_Glyph(face, glyphIndex, FT_LOAD_DEFAULT);
  if (face->glyph->format != FT_GLYPH_FORMAT_BITMAP) {
    FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
  }
  assert(face->glyph->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY);
  return face->glyph;
}

FT_GlyphSlot loadGlyph(const Face& face, char32_t codePoint) {
  return loadGlyph(face.get(), codePoint);
}

FT_GlyphSlot loadGlyph(std::size_t index, char32_t codePoint) {
  return loadGlyph(faces[index].get(), codePoint);
}

int getLineHeight(std::size_t index) {
  return faces[index]->height * config::FONT_NAMES_AND_SIZES[index].pixelHeight / faces[index]->units_per_EM;
}

int getTopToBaseInline(std::size_t index) {
  return faces[index]->ascender * config::FONT_NAMES_AND_SIZES[index].pixelHeight / faces[index]->units_per_EM;
}

} }
