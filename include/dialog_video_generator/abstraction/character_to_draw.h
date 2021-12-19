// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_CHARACTER_TO_DRAW_H
#define DIALOGVIDEOGENERATOR_CHARACTER_TO_DRAW_H

#include <dialog_video_generator/abstraction/character.h>
#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace abstraction {

class CharacterToDraw : public Character, public drawable::Drawable {
public:
  CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson = false);
  CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat,
                  const std::string& standRootDir, const std::string& poseFormat, const std::string& exprFormat,
                  Vec2i bottomCenterOffset, bool firstPerson = false, bool drawStand = true);
  ~CharacterToDraw() override;
  Frames duration() const override;
  size_t bufferCount() const override;
  size_t nextFrame(Frames timeInScene) override;
  void nextScene(bool stop, Frames point) override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

  void keepsAllInNextScene();
  void changesExprInNextScene(const std::string& pose, const std::string& expression, bool flip = false);
  void movesInNextScene(const std::string& pose, const std::string& expression, Vec2i newOffset);
  std::shared_ptr<Drawable> speaksInNextScene(const std::shared_ptr<drawable::TextLike>& lines, Action newAction = Action::NORMAL);
  std::shared_ptr<Drawable> speaksAndChangesExprInNextScene(const std::shared_ptr<drawable::TextLike>& lines,
                                       const std::string& pose, const std::string& expression, bool flip = false,
                                       Action newAction = Action::NORMAL);
  void setOffset(Vec2i offset);

  std::shared_ptr<drawable::Drawable> getDialog();

private:
  std::shared_ptr<Drawable> stand;
  std::shared_ptr<Drawable> dialog;
};

}}

#endif //DIALOGVIDEOGENERATOR_CHARACTER_TO_DRAW_H
