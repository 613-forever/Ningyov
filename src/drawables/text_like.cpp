// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/drawable.h>

#include <tinyutf8/tinyutf8.h>
#include <dialog_video_generator/text/text_render_details.h>

using namespace common613;

namespace dialog_video_generator { namespace drawable {

TextLike::TextLike(const std::string& content, Vec2i pos, Size sz, bool colorType,
                   size_t start, size_t speedNum, size_t speedDen)
    : size(sz), start(start), current(start), speedNum(speedNum), speedDen(speedDen), colorType(colorType), glyphs() {
  tiny_utf8::string buffer(content);

  Vec2i offsetInTextBox{0, 0};
  for (char32_t c : buffer) {
    if (c == '\n') {
      offsetInTextBox.x() = 0;
      offsetInTextBox.y() += 64;
    } else {
      FT_GlyphSlot slot = font::loadGlyph(font::faceForChineseText, c);
      assert(slot->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY);
      const Size glyphSize = Size::of(slot->bitmap.rows, slot->bitmap.width);
      if (glyphSize.total() == 0) {
        BOOST_LOG_TRIVIAL(debug) << fmt::format("Empty glyph for {:#x}, {}x{}.", c, glyphSize.h(), glyphSize.w());
      } else {
        BOOST_LOG_TRIVIAL(trace)
          << fmt::format("Generating glyph bitmap for {:#x}, {}x{}.", c, glyphSize.h(), glyphSize.w());
        Memory memory(glyphSize.total());
        if (slot->bitmap.pitch > 0) {
          unsigned char* memoryPtr = memory.data(), * bufferPtr = slot->bitmap.buffer;
          for (int i = 0; i < glyphSize.h(); ++i) {
            std::memcpy(memoryPtr, bufferPtr, glyphSize.w());
            memoryPtr += glyphSize.w();
            bufferPtr += slot->bitmap.pitch;
          }
        } else {
          unsigned char* memoryPtr = memory.data() + (glyphSize.h() - 1) * glyphSize.w(),
              * bufferPtr = slot->bitmap.buffer;
          for (int i = 0; i < glyphSize.h(); ++i) {
            std::memcpy(memoryPtr, bufferPtr, glyphSize.w());
            memoryPtr -= glyphSize.w();
            bufferPtr += slot->bitmap.pitch;
          }
        }
//      for (int i = 0; i < glyphSize.h(); ++i) {
//        for (int j = 0; j < glyphSize.w(); ++j) {
//          std::printf("%02x ", (int)memory[i * glyphSize.w() + j]);
//        }
//        std::printf("\n");
//      }
//      std::printf("---\n");
        CudaMemory cudaMemory = cuda::copyFromCPUMemory(memory);
        if (offsetInTextBox.x() + (slot->metrics.width >> 6) > size.w()) {
          offsetInTextBox.x() = 0;
          offsetInTextBox.y() += 64;
        }
        glyphs.emplace_back(Image{RawImage{glyphSize, cudaMemory}, 1, {
            checked_cast<Dim>(pos.x() + offsetInTextBox.x() + slot->bitmap_left),
            checked_cast<Dim>(pos.y() + offsetInTextBox.y() - slot->bitmap_top)
        }});
      }
      offsetInTextBox.x() = checked_cast<Dim>((slot->advance.x >> 6) + offsetInTextBox.x());
    }
  }
}

TextLike::~TextLike() = default;

Frames TextLike::duration() const {
  return Frames::of((speedDen * (glyphs.size() - start) + speedNum - 1) / speedNum + 1);
}

std::size_t TextLike::bufferCount() const {
  return glyphs.size();
}

std::size_t TextLike::nextFrame(Frames timeInShot) {
  std::size_t last = current;
  if (last < glyphs.size()) {
    current = start + (timeInShot.x() * speedNum + speedDen - 1) / speedDen;
    if (current > glyphs.size()) {
      current = glyphs.size();
    }
    return current > last ? glyphs.size() - last : 0;
  } else {
    return 0;
  }
}

void TextLike::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  BOOST_LOG_TRIVIAL(trace) << fmt::format("TextLike adds tasks for {} glyphs (total {}).", current, glyphs.size());
  for (size_t i = 0; i < glyphs.size(); ++i) {
    tasks.emplace_back(DrawTask{
        glyphs[i].pos.y() + offset.y(), glyphs[i].pos.x() + offset.x(),
        glyphs[i].raw.size.h(), glyphs[i].raw.size.w(),
        1, alpha,
        true, true, colorType, (alpha < 16),
        (i >= current), false, false, false,
        glyphs[i].raw.memory.get()
    });
  }
}

} }
