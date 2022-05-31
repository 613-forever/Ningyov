// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/abstraction/character.h>

#include <dialog_video_generator/abstraction/action_animations.h>
#include <dialog_video_generator/abstraction/dialog.h>
#include <dialog_video_generator/drawables/stand.h>

#include <utility>

using namespace dialog_video_generator::drawable;

namespace dialog_video_generator {
namespace config {
std::uint16_t STAND_MULTIPLIER = 4;
}

namespace abstraction {

Character::Character(std::string dialogDir, std::string dialogFormat, bool firstPerson /* = false */) :
    hasStandToDraw(false), drawStand(false), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{}, actionAnimated{true}, offset{},
    dlgDir(std::move(dialogDir)), dlgFmt(std::move(dialogFormat)) {
  clearDialogResources(isFirstPerson);
}

Character::Character(std::string dialogDir, std::string dialogFormat,
                     std::string standRootDir, std::string characterString,
                     Vec2i bottomCenterOffset, bool firstPerson /* = isFalse */, bool drawStand /* = true */) :
    hasStandToDraw(true), drawStand(drawStand), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{Action::NORMAL}, actionAnimated{true}, offset{bottomCenterOffset},
    dlgDir(std::move(dialogDir)), dlgFmt(std::move(dialogFormat)),
    standDir(std::move(standRootDir)), charStr(std::move(characterString)) {
  clearDialogResources(isFirstPerson);
  // initEyeBinder();
  blinkSelector = std::make_shared<BlinkSelector>(&eyeBinder);
  BlinkSelector::reset(blinkSelector.get());
}

Character::~Character() = default;

void Character::setAction(Action newAction) {
  if (newAction == Action::NO_CHANGE) {
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
  isFirstPerson = firstPerson;
}

std::shared_ptr<Texture> Character::getNormalDialog() {
  if (!dialog) {
    dialog = std::make_shared<Texture>(dlgDir, dlgFmt + (isFirstPerson ? FIRST_PERSON : "") + ACT_NORMAL);
  }
  return dialog;
}

std::shared_ptr<Texture> Character::getThinkingDialog() {
  if (!thinkingDialog) {
    thinkingDialog = std::make_shared<Texture>(dlgDir, dlgFmt + (isFirstPerson ? FIRST_PERSON : "") + ACT_THINKING);
  }
  return thinkingDialog;
}

std::shared_ptr<Texture> Character::getShoutingDialog() {
  if (!shoutingDialog) {
    shoutingDialog = std::make_shared<Texture>(dlgDir, dlgFmt + (isFirstPerson ? FIRST_PERSON : "") + ACT_SHOUTING);
  }
  return shoutingDialog;
}

std::shared_ptr<Texture> Character::getMurmuringDialog() {
  if (!murmuringDialog) {
    murmuringDialog = std::make_shared<Texture>(dlgDir, dlgFmt + (isFirstPerson ? FIRST_PERSON : "") + ACT_MURMURING);
  }
  return murmuringDialog;
}

void Character::setOffset(Vec2i newOffset) {
  offset = newOffset;
}

void Character::keepsAllInNextShot() {
  stand->nextShot(false, 0_fr);
  actionAnimated = true;
}

void Character::changesExprInNextShot(const std::string& pose, const std::string& expression, bool flip/* = false*/) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  stand = std::make_shared<Stand>(
      standDir, charStr, pose, expression,
      config::STAND_MULTIPLIER, 0_fr, blinkSelector, flip
  );
  actionAnimated = true;
}

void Character::movesInNextShot(const std::string& pose, const std::string& expression, Vec2i newOffset) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  // TODO: movement animation
}

void Character::speaksInNextShot(const std::shared_ptr<drawable::TextLike>& lines, Action newAction/*= NORMAL*/,
                                 bool addMouthAnimation/* = true*/) {
  if (hasStandToDraw && drawStand) {
    COMMON613_REQUIRE(stand != nullptr, "Setting character speaking before expression.");
    if (addMouthAnimation) {
      stand->setSpeakingDuration(lines->duration());
    }
  }
  setAction(newAction);
}

void Character::speaksAndChangesExprInNextShot(const std::shared_ptr<drawable::TextLike>& lines,
                                               const std::string& pose, const std::string& expression,
                                               bool flip/* = false*/,
                                               Action newAction/* = Action::NORMAL*/,
                                               bool addMouthAnimation/* = true*/) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without any stand CG.");
  stand = std::make_shared<Stand>(
      standDir, charStr, pose, expression,
      config::STAND_MULTIPLIER, addMouthAnimation ? lines->duration() : 0_fr, blinkSelector, flip
  );
  setAction(newAction);
}

std::shared_ptr<Drawable> Character::getStand() {
  if (!drawStand) return nullptr;
  COMMON613_REQUIRE(hasStandToDraw, "Getting stand CG for a character without any stand CG.");
  auto translated = translate(stand, offset);
  if (actionAnimated) {
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

std::shared_ptr<Drawable> Character::getDialog(const std::shared_ptr<TextLike>& lines) {
  switch (action) {
  case Action::THINKING:
    return makeDialog(getThinkingDialog(), lines);
  case Action::SHOUTING: {
    auto dialogWithText = makeDialog(getShoutingDialog(), lines);
    if (actionAnimated) {
      return dialogWithText;
    } else {
      return animateShouting(dialogWithText);
    }
  }
  case Action::MURMURING: {
    auto dialogWithText = makeDialog(getMurmuringDialog(), lines);
    if (actionAnimated) {
      return dialogWithText;
    } else {
      return animateMurmuring(dialogWithText);
    }
  }
  default:
    return makeDialog(getNormalDialog(), lines);
  }
}

// Using new object is preferred.
void Character::nextAct(bool firstPerson /*= false*/) {
  clearDialogResourcesIfChanged(firstPerson);
}

}
}
