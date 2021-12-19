// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/character_to_draw.h>

namespace dialog_video_generator { namespace abstraction {

CharacterToDraw::CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson)
    : Character(dialogDir, dialogFormat, firstPerson) {}

CharacterToDraw::CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat,
                                 const std::string& standRootDir,
                                 const std::string& poseFormat, const std::string& exprFormat,
                                 Vec2i bottomCenterOffset, bool firstPerson, bool drawStand)
    : Character(dialogDir, dialogFormat,
                standRootDir, poseFormat, exprFormat,
                bottomCenterOffset, firstPerson, drawStand) {}

CharacterToDraw::~CharacterToDraw() = default;

Frames CharacterToDraw::duration() const {
  assert(stand != nullptr); // hidden but asked to be displayed?
  return stand->duration();
}

size_t CharacterToDraw::bufferCount() const {
  assert(stand != nullptr); // hidden but asked to be displayed?
  return stand->bufferCount();
}

size_t CharacterToDraw::nextFrame(Frames timeInScene) {
  assert(stand != nullptr); // hidden but asked to be displayed?
  return stand->nextFrame(timeInScene);
}

void CharacterToDraw::nextScene(bool stop, Frames point) {
  assert(stand != nullptr); // hidden but asked to be displayed?
  stand->nextScene(stop, point);
}

void CharacterToDraw::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  assert(stand != nullptr); // hidden but asked to be displayed?
  stand->addTask(offset, alpha, tasks);
}

void CharacterToDraw::keepsAllInNextScene() {
  Character::keepsAllInNextScene();
  stand = getStand();
  dialog = nullptr;
}

void CharacterToDraw::changesExprInNextScene(const std::string& pose, const std::string& expression, bool flip) {
  Character::changesExprInNextScene(pose, expression, flip);
  stand = getStand();
  dialog = nullptr;
}

void CharacterToDraw::movesInNextScene(const std::string& pose, const std::string& expression, Vec2i newOffset) {
  Character::movesInNextScene(pose, expression, newOffset);
  stand = getStand();
  dialog = nullptr;
}

std::shared_ptr<Drawable> CharacterToDraw::speaksInNextScene(const std::shared_ptr<drawable::TextLike>& lines, Action newAction) {
  Character::speaksInNextScene(lines, newAction);
  stand = getStand();
  dialog = Character::getDialog(lines);
  return dialog;
}

std::shared_ptr<Drawable> CharacterToDraw::speaksAndChangesExprInNextScene(const std::shared_ptr<drawable::TextLike>& lines,
                                                      const std::string& pose,
                                                      const std::string& expression,
                                                      bool flip,
                                                      Action newAction) {
  Character::speaksAndChangesExprInNextScene(lines, pose, expression, flip, newAction);
  stand = getStand();
  dialog = Character::getDialog(lines);
  return dialog;
}

void CharacterToDraw::setOffset(Vec2i offset) {
  Character::setOffset(offset);
  stand = getStand();
}

std::shared_ptr<drawable::Drawable> CharacterToDraw::getDialog() {
  return dialog;
}

}}
