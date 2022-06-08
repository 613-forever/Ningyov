// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#pragma once
#ifndef NINGYOV_CHARACTER_TO_DRAW_H
#define NINGYOV_CHARACTER_TO_DRAW_H

#include <ningyov/abstraction/character.h>
#include <ningyov/drawable.h>

namespace ningyov { namespace abstraction {

class CharacterToDraw : public Character, public Drawable {
public:
  CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson = false);
  CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat,
                  const std::string& standRootDir, const std::string& characterString,
                  Vec2i bottomCenterOffset, bool firstPerson = false, bool drawStand = true);
  ~CharacterToDraw() override;
  Frames duration() const override;
  size_t bufferCount() const override;
  size_t nextFrame(Frames timeInShot) override;
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

  void keepsAllInNextShot();
  void changesExprInNextShot(const std::string& pose, const std::string& expression, bool flip = false);
  void movesInNextShot(const std::string& pose, const std::string& expression, Vec2i newOffset);
  std::shared_ptr<Drawable> speaksInNextShot(const std::shared_ptr<drawable::TextLike>& lines, Action newAction = Action::NORMAL,
                                             bool addMouthAnimation = true);
  std::shared_ptr<Drawable> speaksAndChangesExprInNextShot(const std::shared_ptr<drawable::TextLike>& lines,
                                       const std::string& pose, const std::string& expression, bool flip = false,
                                       Action newAction = Action::NORMAL, bool addMouthAnimation = true);
  void setOffset(Vec2i offset);

  std::shared_ptr<Drawable> getDialog();

private:
  std::shared_ptr<Drawable> stand;
  std::shared_ptr<Drawable> dialog;
};

}}

#endif //NINGYOV_CHARACTER_TO_DRAW_H
