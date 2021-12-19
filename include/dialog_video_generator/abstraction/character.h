// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#ifndef DIALOGVIDEOGENERATOR_CHARACTER_H
#define DIALOGVIDEOGENERATOR_CHARACTER_H

#include <memory>
#include <cstdint>
#include <dialog_video_generator/math/pos_utils.h>
#include <dialog_video_generator/math/time_utils.h>

namespace dialog_video_generator {
namespace drawable {
class Drawable;
class Texture;
class Stand;
class TextLike;
}

namespace config {
extern std::uint16_t STAND_MULTIPLIER;
}

namespace abstraction {

enum class Action {
  NO_CHANGE = -1,
  NORMAL = 0,
  THINKING,
  SHOUTING,
  MURMURING,
};

class Character {
public:
  static constexpr const char* const ACT_NORMAL = "";
  static constexpr const char* const ACT_THINKING = "_th";
  static constexpr const char* const ACT_SHOUTING = "_sh";
  static constexpr const char* const ACT_MURMURING = "_mur";
  static constexpr const char* const FIRST_PERSON = "_1st";

  // non-display character
  Character(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson = false);
  // display character. use CharacterToDraw.
  Character(const std::string& dialogDir, const std::string& dialogFormat,
            const std::string& standRootDir, const std::string& poseFormat, const std::string& exprFormat,
            Vec2i bottomCenterOffset, bool firstPerson = false, bool drawStand = true);
  ~Character();

private:
  void setAction(Action newAction);
  void clearDialogResources(bool firstPerson);
  void clearDialogResourcesIfChanged(bool firstPerson) {
    if (firstPerson != isFirstPerson) clearDialogResources(firstPerson);
  }
  void initEyeBinder();

  std::shared_ptr<drawable::Texture> getNormalDialog();
  std::shared_ptr<drawable::Texture> getThinkingDialog();
  std::shared_ptr<drawable::Texture> getShoutingDialog();
  std::shared_ptr<drawable::Texture> getMurmuringDialog();

public:
  void setOffset(Vec2i newOffset);

  void keepsAllInNextScene();
  void changesExprInNextScene(const std::string& pose, const std::string& expression, bool flip = false);
  void movesInNextScene(const std::string& pose, const std::string& expression, Vec2i newOffset);
  void speaksInNextScene(const std::shared_ptr<drawable::TextLike>& lines, Action newAction = Action::NORMAL);
  void speaksAndChangesExprInNextScene(const  std::shared_ptr<drawable::TextLike>& lines,
                                       const std::string& pose, const std::string& expression, bool flip = false,
                                       Action newAction = Action::NORMAL);

  std::shared_ptr<drawable::Drawable> getStand();
  std::shared_ptr<drawable::Drawable> getDialog(const std::shared_ptr<drawable::TextLike>& speaking);

  void nextAct(bool firstPerson = false); // next act, clear FP flags

private:
  // status
  bool hasStandToDraw, drawStand;
  bool isFirstPerson;
  Frames eyeBinder;
  std::string dlgDir, standDir, dlgFmt;
  // unlikely to change ( changes only when first-person changes )
  std::shared_ptr<drawable::Texture> dialog, thinkingDialog, shoutingDialog, murmuringDialog;
  std::shared_ptr<drawable::Stand> stand{};
  // scene-wise
  Action action;
  bool actionAnimated; // true: depress action animation. useful when the scene follows another which has animated it.
  Vec2i offset;
  std::string poseFmt, exprFmt;
};

} }

#endif //DIALOGVIDEOGENERATOR_CHARACTER_H
