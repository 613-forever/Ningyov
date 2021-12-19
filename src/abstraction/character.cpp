// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/character.h>

#include <dialog_video_generator/abstraction/action_animation.h>
#include <dialog_video_generator/abstraction/dialog.h>

using namespace dialog_video_generator::drawable;

namespace dialog_video_generator {
namespace config {
std::uint16_t STAND_MULTIPLIER = 4;
}

namespace abstraction {

Character::Character(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson /* = false */) :
    hasStandToDraw(false), drawStand(false), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{}, actionAnimated{true}, offset{},
    dlgDir(dialogDir), dlgFmt(dialogFormat), poseFmt{}, exprFmt{} {
  clearDialogResources(isFirstPerson);
}

Character::Character(const std::string& dialogDir, const std::string& dialogFormat,
                     const std::string& standRootDir, const std::string& poseFormat, const std::string& exprFormat,
                     Vec2i bottomCenterOffset, bool firstPerson /* = isFalse */, bool drawStand /* = true */) :
    hasStandToDraw(true), drawStand(drawStand), isFirstPerson(firstPerson), eyeBinder{0_fr},
    action{Action::NORMAL}, actionAnimated{true}, offset{bottomCenterOffset},
    dlgDir(dialogDir), dlgFmt(dialogFormat), standDir(standRootDir), poseFmt(poseFormat), exprFmt(exprFormat) {
  clearDialogResources(isFirstPerson);
  initEyeBinder();
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
}

void Character::initEyeBinder() {
  Stand::refreshEyeBlinkCountDown(&eyeBinder);
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

void Character::keepsAllInNextScene() {
  stand->nextScene(false, 0_fr);
  actionAnimated = true;
}

void Character::changesExprInNextScene(const std::string& pose, const std::string& expression, bool flip/* = false*/) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  stand = std::make_shared<Stand>(
      standDir, fmt::format(poseFmt, pose), fmt::format(exprFmt, expression),
      config::STAND_MULTIPLIER, 0_fr, flip
  );
  stand->bindEyeStatus(&eyeBinder);
  actionAnimated = true;
}

void Character::movesInNextScene(const std::string& pose, const std::string& expression, Vec2i moveTo) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without stand CG.");
  // TODO: move animation
}

void Character::speaksInNextScene(const std::shared_ptr<drawable::TextLike>& lines, Action newAction/*= NORMAL*/) {
  if (hasStandToDraw && drawStand) {
    COMMON613_REQUIRE(stand != nullptr, "Setting character speaking before expression.");
    stand->setSpeakingDuration(lines->duration());
  }
  setAction(newAction);
}

void Character::speaksAndChangesExprInNextScene(const std::shared_ptr<drawable::TextLike>& lines,
                                                const std::string& pose, const std::string& expression,
                                                bool flip/* = false*/,
                                                Action newAction /*= Action::NORMAL*/) {
  COMMON613_REQUIRE(hasStandToDraw, "Setting stand information for a character without any stand CG.");
  stand = std::make_shared<Stand>(
      standDir, fmt::format(poseFmt, pose), fmt::format(exprFmt, expression),
      config::STAND_MULTIPLIER, lines->duration(), flip
  );
  stand->bindEyeStatus(&eyeBinder);
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
