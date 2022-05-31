// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/abstraction/character_to_draw.h>

namespace dialog_video_generator {
namespace abstraction {

CharacterToDraw::CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson)
    : Character(dialogDir, dialogFormat, firstPerson) {}

CharacterToDraw::CharacterToDraw(const std::string& dialogDir, const std::string& dialogFormat,
                                 const std::string& standRootDir, const std::string& characterString,
                                 Vec2i bottomCenterOffset, bool firstPerson, bool drawStand)
    : Character(dialogDir, dialogFormat,
                standRootDir, characterString,
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

size_t CharacterToDraw::nextFrame(Frames timeInShot) {
  assert(stand != nullptr); // hidden but asked to be displayed?
  return stand->nextFrame(timeInShot);
}

std::shared_ptr<drawable::Drawable> CharacterToDraw::nextShot(bool stop, Frames point) {
  assert(stand != nullptr); // hidden but asked to be displayed?
  stand->nextShot(stop, point);
  return shared_from_this();
}

void CharacterToDraw::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  assert(stand != nullptr); // hidden but asked to be displayed?
  stand->addTask(offset, alpha, tasks);
}

void CharacterToDraw::keepsAllInNextShot() {
  Character::keepsAllInNextShot();
  stand = getStand();
  dialog = nullptr;
}

void CharacterToDraw::changesExprInNextShot(const std::string& pose, const std::string& expression, bool flip) {
  Character::changesExprInNextShot(pose, expression, flip);
  stand = getStand();
  dialog = nullptr;
}

void CharacterToDraw::movesInNextShot(const std::string& pose, const std::string& expression, Vec2i newOffset) {
  Character::movesInNextShot(pose, expression, newOffset);
  stand = getStand();
  dialog = nullptr;
}

std::shared_ptr<Drawable> CharacterToDraw::speaksInNextShot(const std::shared_ptr<drawable::TextLike>& lines,
                                                            Action newAction/* = Action::NORMAL*/,
                                                            bool addMouthAnimation/* = true*/) {
  Character::speaksInNextShot(lines, newAction, addMouthAnimation);
  stand = getStand();
  dialog = Character::getDialog(lines);
  return dialog;
}

std::shared_ptr<Drawable> CharacterToDraw::speaksAndChangesExprInNextShot(const std::shared_ptr<drawable::TextLike>& lines,
                                                                          const std::string& pose,
                                                                          const std::string& expression,
                                                                          bool flip/* = false*/,
                                                                          Action newAction/* = Action::NORMAL*/,
                                                                          bool addMouthAnimation/* = true*/) {
  Character::speaksAndChangesExprInNextShot(lines, pose, expression, flip, newAction, addMouthAnimation);
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

}
}
