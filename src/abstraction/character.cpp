// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/character.h>

#include <dialog_video_generator/drawable.h>

using namespace dialog_video_generator::drawable;

namespace dialog_video_generator {
namespace config {
std::uint16_t STAND_MULTIPLIER = 4;
}

namespace abstraction {

Character::Character(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson /* = false */) :
    hasStandToDraw(false), drawStand(false), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{}, actionAnimated{true}, offset{},
    dlgFmt(dialogFormat), poseFmt{}, exprFmt{} {
  clearDialogResources(isFirstPerson);
}

Character::Character(const std::string& dialogDir, const std::string& dialogFormat,
                     const std::string& standRootDir, const std::string& poseFormat, const std::string& exprFormat,
                     Vec2i bottomCenterOffset, bool firstPerson /* = isFalse */, bool drawStand /* = true */) :
    hasStandToDraw(true), drawStand(drawStand), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{Action::NORMAL}, actionAnimated{true}, offset{bottomCenterOffset},
    dlgDir(dialogDir), standDir(standRootDir),
    dlgFmt(dialogFormat), poseFmt(poseFormat), exprFmt(exprFormat) {
  clearDialogResources(isFirstPerson);
  initEyeBinder();
}

Character::~Character() = default;

void Character::nextScene() {}

void Character::setAction(Action newAction) {
  if (action == Action::NO_CHANGE) {
    actionAnimated = true;
  } else {
    actionAnimated = false;
    action = newAction;
  }
}

void Character::clearDialogResources(bool firstPerson) {
  dialog = nullptr;
  thinkingDialog = nullptr;
  shoutingDialog = nullptr;
  murmuringDialog = nullptr;
}

void Character::initEyeBinder() {
  Stand::refreshEyeBlinkCountDown(&eyeBinder);
}

std::shared_ptr<drawable::Texture> Character::getNormalDialog() {
  if (!dialog) {
    dialog = std::make_shared<Texture>(dlgDir, dlgFmt + ACTION_NORMAL + FIRST_PERSON);
  }
  return dialog;
}

std::shared_ptr<drawable::Texture> Character::getThinkingDialog() {
  if (!dialog) {
    dialog = std::make_shared<Texture>(dlgDir, dlgFmt + ACTION_THINKING + FIRST_PERSON);
  }
  return dialog;
}

std::shared_ptr<drawable::Texture> Character::getShoutingDialog() {
  if (!dialog) {
    dialog = std::make_shared<Texture>(dlgDir, dlgFmt + ACTION_SHOUTING + FIRST_PERSON);
  }
  return dialog;
}

std::shared_ptr<drawable::Texture> Character::getMurmuringDialog() {
  if (!dialog) {
    dialog = std::make_shared<Texture>(dlgDir, dlgFmt + ACTION_MURMURING + FIRST_PERSON);
  }
  return dialog;
}

void Character::keepsAllInNextScene() {
  nextScene();
  stand->nextScene(false, 0_fr);
  actionAnimated = true;
}

void Character::changesExprInNextScene(const std::string& pose, const std::string& expression, bool flip) {
  nextScene();
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  stand = std::make_shared<Stand>(
      standDir, fmt::format(poseFmt, pose), fmt::format(exprFmt, expression),
      config::STAND_MULTIPLIER, 0_fr, flip
  );
  stand->bindEyeStatus(&eyeBinder);
  actionAnimated = true;
}

void Character::movesInNextScene(const std::string& pose, const std::string& expression, Vec2i moveTo) {
  nextScene();
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  // TODO: move animation
}

void Character::speaksInNextScene(const TextLike& lines, Action newAction /*= Action::NO_CHANGE*/) {
  nextScene();
  if (hasStandToDraw && drawStand) {
    COMMON613_REQUIRE(stand != nullptr, "Setting character speaking before expression.");
    stand->setSpeakingDuration(lines.duration());
  }
  setAction(newAction);
}

void Character::speaksAndChangesExprInNextScene(const std::string& pose, const std::string& expression, bool flip,
                                                const TextLike& lines, Action newAction /*= Action::NO_CHANGE*/) {
  nextScene();
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without any stand CG.");
  stand = std::make_shared<Stand>(
      standDir, fmt::format(poseFmt, pose), fmt::format(exprFmt, expression),
      config::STAND_MULTIPLIER, lines.duration(), flip
  );
  stand->bindEyeStatus(&eyeBinder);
  setAction(newAction);
}

std::shared_ptr<Drawable> Character::getStand() {
  if (!drawStand) return nullptr;
  COMMON613_REQUIRE(hasStandToDraw, "Getting stand CG for a character without any stand CG.");
  auto translated = translate(stand, offset);
  if (!actionAnimated) {
    return translated;
  }
  switch (action) {
  case Action::SHOUTING:
    return animateShouting(translated);
  case Action::MURMURING:
    return animateMurmuring(translated);
  default:
    return translated;
  }
}

std::shared_ptr<Drawable> Character::getSpeakingDialog(std::shared_ptr<Drawable>& speaking) {
  switch (action) {
  case Action::THINKING:
    return getThinkingDialog();
  case Action::SHOUTING:
    if (!actionAnimated) {
      speaking = animateShouting(speaking);
    }
    return getShoutingDialog();
  case Action::MURMURING:
    if (!actionAnimated) {
      speaking = animateMurmuring(speaking);
    }
    return getMurmuringDialog();
  default:
    return getNormalDialog();
  }
}

void Character::nextAct(bool firstPerson /*= false*/) {
  clearDialogResourcesIfChanged(firstPerson);
  nextScene();
}

} }