// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/abstraction/dialog.h>

namespace ningyov { namespace abstraction {

Dialog::Dialog(std::shared_ptr<drawable::Texture> dialog, std::shared_ptr<drawable::TextLike> text)
: dialog(std::move(dialog)), text(std::move(text)) {}

Dialog::~Dialog() = default;

Frames Dialog::duration() const {
  return text->duration();
}

std::size_t Dialog::bufferCount() const {
  return dialog->bufferCount() + text->bufferCount();
}

std::size_t Dialog::nextFrame(Frames timeInShot) {
  std::size_t dialogUpdate = dialog->nextFrame(timeInShot);
  std::size_t textUpdate = text->nextFrame(timeInShot);
  return dialogUpdate > 0 ? dialogUpdate + textUpdate : textUpdate;
}

std::shared_ptr<Drawable> Dialog::nextShot(bool stop, Frames point) {
  dialog->nextShot(stop, point);
  text->nextShot(stop, point);
  return shared_from_this();
}

void Dialog::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  dialog->addTask(offset, alpha, tasks);
  text->addTask(offset, alpha, tasks);
}

} }
