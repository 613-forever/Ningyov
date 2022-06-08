// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/drawable.h>

#include <tinyutf8/tinyutf8.h>
#include <dialog_video_generator/text/text_render_details.h>

using namespace common613;

namespace dialog_video_generator { namespace drawable {

TextLike::TextLike(const std::string& content, Vec2i pos, Size sz, bool colorType,
                   std::size_t start, std::size_t speedNum, std::size_t speedDen, std::size_t fontIndex)
    : size(sz), current(-1), color(colorType ? /*config::COLOR_THINKING*/Color4b::of(255, 255, 255, 255) : /*config::COLOR_SPEAKING)*/Color4b::of(0, 0, 0, 255)), partial(), indices() {
  tiny_utf8::string buffer(content);

  // calc indices size (duration)
  auto updateFrames = (speedDen * (buffer.length() - start) + speedNum - 1) / speedNum + 1;
  auto updateTimes = ((buffer.length() - start) + speedNum - 1) / speedNum + 1;
  CudaMemory partialBuffersOnGPU;
  {
    std::vector<unsigned char*> partialBuffers;
    partial.reserve(updateTimes);
    for (int i = 0; i < updateTimes; ++i) {
      CudaMemory memory = cuda::allocateMemory(size.h(), size.w());
      partial.emplace_back(Image{
        RawImage{size, memory},
        1,
        pos,
        false,
      });
      partialBuffers.push_back(memory.get());
    }
    partialBuffersOnGPU = cuda::copyFromCPUMemory(partialBuffers.data(), partialBuffers.size() * sizeof(unsigned char*));
  }

  // init glyphs
  std::vector<Image> glyphs;
  auto lineHeight = static_cast<Dim>(font::getLineHeight(fontIndex)); // unchecked
  // max if multiple fonts are allowed in the future
  {
    Vec2i offsetInTextBox{0, 0};
    offsetInTextBox.y() += font::getTopToBaseInline(fontIndex);
    for (char32_t c: buffer) {
      if (c == '\n') {
        offsetInTextBox.x() = 0;
        offsetInTextBox.y() += lineHeight;
      } else {
        FT_GlyphSlot slot = font::loadGlyph(fontIndex, c);
        assert(slot->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY);
        const Size glyphSize = Size::of(slot->bitmap.rows, slot->bitmap.width);
        if (glyphSize.total() == 0) {
          BOOST_LOG_TRIVIAL(debug) << fmt::format("Empty glyph for {:#x}, {}x{}.", c, glyphSize.h(), glyphSize.w());
        } else {
          BOOST_LOG_TRIVIAL(trace) << fmt::format("Generating glyph bitmap for {:#x}, {}x{}.", c, glyphSize.h(), glyphSize.w());
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
//          for (int i = 0; i < glyphSize.h(); ++i) {
//            for (int j = 0; j < glyphSize.w(); ++j) {
//              std::printf("%02x ", (int)memory[i * glyphSize.w() + j]);
//            }
//            std::printf("\n");
//          }
//          std::printf("---\n");
          CudaMemory cudaMemory = cuda::copyFromCPUMemory(memory);
          if (offsetInTextBox.x() + (slot->metrics.width >> 6) > size.w()) {
            offsetInTextBox.x() = 0;
            offsetInTextBox.y() += lineHeight; // line height
          }
          // baseline height
          glyphs.emplace_back(Image{RawImage{glyphSize, cudaMemory}, 1, {
              checked_cast<Dim>(offsetInTextBox.x() + slot->bitmap_left),
              checked_cast<Dim>(offsetInTextBox.y() - slot->bitmap_top)
          }});
        }
        offsetInTextBox.x() = checked_cast<Dim>((slot->advance.x >> 6) + offsetInTextBox.x());
      }
    }
  }

  // make updates for frames
  std::vector<TextTask> tasks;
  {
    indices.resize(updateFrames);

    std::size_t last = start;
    std::size_t bufferId = 0;
    // frame 0
    for (auto glyphIndex = 0u; glyphIndex < start; ++glyphIndex) {
      auto& glyph = glyphs[glyphIndex];
      tasks.emplace_back(TextTask{
          glyph.pos.y(), glyph.pos.x(),
          glyph.raw.size.h(), glyph.raw.size.w(),
          color.r(), color.g(), color.b(), true,
          glyph.raw.memory.get()
      });
    }
    indices[0] = bufferId++;
    // frame 1 - updates-1
    for (auto index = 1u; index < updateFrames; ++index) {
      if (last >= glyphs.size()) {
        break;
      }
      current = start + ((index - 1) * speedNum + speedDen - 1) / speedDen;
      if (current >= glyphs.size()) {
        current = glyphs.size();
      }
      if (current == last) {
        indices[index] = bufferId;
        continue;
      }
      auto firstTaskForFrame = tasks.size();
      for (auto glyphIndex = last; glyphIndex < current; ++glyphIndex) {
        auto& glyph = glyphs[glyphIndex];
        tasks.emplace_back(TextTask{
            glyph.pos.y(), glyph.pos.x(),
            glyph.raw.size.h(), glyph.raw.size.w(),
            color.r(), color.g(), color.b(), true,
            glyph.raw.memory.get()
        });
      }
      tasks[firstTaskForFrame].reuse = false;
      indices[index] = bufferId++;
      last = current;
    }
  }

  assert(partial.size() == std::count_if(tasks.begin(), tasks.end(), [](const TextTask& task) { return !task.reuse; }) + 1);
  auto tasksOnGPU = cuda::copyFromCPUMemory(tasks.data(), tasks.size() * sizeof(TextTask));
  cuda::mergeGlyphMasks(reinterpret_cast<unsigned char**>(partialBuffersOnGPU.get()),
                        reinterpret_cast<TextTask*>(tasksOnGPU.get()), tasks.size(),
                        size.h(), size.w());
//  cuda::checkCudaError();
}

TextLike::~TextLike() = default;

Frames TextLike::duration() const {
  return Frames::of(indices.size());
}

std::size_t TextLike::bufferCount() const {
  return 1;
}

std::size_t TextLike::nextFrame(Frames timeInShot) {
  auto last = current;
  current = timeInShot.x() < indices.size() ? indices[timeInShot.x()] : indices.back();
  return (current != last);
}

std::shared_ptr<Drawable> TextLike::nextShot(bool stop, Frames point) {
  if (stop || current >= indices.size()) {
    return std::make_shared<Texture>(partial[indices.back()]);
  }
  indices.erase(indices.begin(), indices.begin() + std::ptrdiff_t(current));
  current = 0;
}

void TextLike::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  partial[current].addTask(offset, true, alpha, false, tasks);
}

} }
