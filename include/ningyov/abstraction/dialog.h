// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#pragma once
#ifndef NINGYOV_DIALOG_H
#define NINGYOV_DIALOG_H

#include <ningyov/drawable.h>

namespace ningyov { namespace abstraction {

class Dialog : public Drawable {
public:
  Dialog(std::shared_ptr<drawable::Texture> dialog, std::shared_ptr<drawable::TextLike> text);
  ~Dialog() override;
  Frames duration() const override;
  std::size_t bufferCount() const override;
  std::size_t nextFrame(Frames timeInShot) override;
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override;
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

#endif //NINGYOV_DIALOG_H
