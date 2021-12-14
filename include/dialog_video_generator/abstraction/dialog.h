// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_DIALOG_H
#define DIALOGVIDEOGENERATOR_DIALOG_H

#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace abstraction {

class Dialog : public Drawable {
public:
  Dialog(std::shared_ptr<drawable::Texture> dialog, std::shared_ptr<drawable::TextLike> text);
  ~Dialog() override;
  Frames duration() const override;
  std::size_t bufferCount() const override;
  std::size_t nextFrame(Frames timeInScene) override;
  void nextScene(bool stop, Frames point) override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
private:
  std::shared_ptr<drawable::Texture> dialog;
  std::shared_ptr<drawable::TextLike> text;
};

inline std::shared_ptr<Dialog> makeDialog(const std::shared_ptr<drawable::Texture>& dialog,
                                          const std::shared_ptr<drawable::TextLike>& text) {
  return std::make_shared<Dialog>(dialog, text);
}

} }

#endif //DIALOGVIDEOGENERATOR_DIALOG_H
