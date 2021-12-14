// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/dialog.h>

namespace dialog_video_generator { namespace abstraction {

Dialog::Dialog(std::shared_ptr<drawable::Texture> dialog, std::shared_ptr<drawable::TextLike> text)
: dialog(std::move(dialog)), text(std::move(text)) {}

Dialog::~Dialog() = default;

Frames Dialog::duration() const {
  return text->duration();
}

std::size_t Dialog::bufferCount() const {
  return dialog->bufferCount() + text->bufferCount();
}

std::size_t Dialog::nextFrame(Frames timeInScene) {
  std::size_t dialogUpdate = dialog->nextFrame(timeInScene);
  std::size_t textUpdate = text->nextFrame(timeInScene);
  return dialogUpdate > 0 ? dialogUpdate + textUpdate : textUpdate;
}

void Dialog::nextScene(bool stop, Frames point) {
  dialog->nextScene(stop, point);
  text->nextScene(stop, point);
}

void Dialog::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  dialog->addTask(offset, alpha, tasks);
  text->addTask(offset, alpha, tasks);
}

} }
