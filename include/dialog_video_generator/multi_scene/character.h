// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#ifndef DIALOGVIDEOGENERATOR_CHARACTER_H
#define DIALOGVIDEOGENERATOR_CHARACTER_H

#include <memory>
#include <cstdint>
#include <dialog_video_generator/math/pos_utils.h>

namespace dialog_video_generator {
namespace drawable {
class Texture;
class Stand;
class TextLike;
}

namespace config {
extern std::uint16_t STAND_MULTIPLIER;
}

namespace multi_scene {

class Character {
public:
  enum class Action {
    NORMAL,
    THINKING,
    SHOUTING,
    MURMURING,
  };
  static constexpr const char* const ACTION_NORMAL = "normal";
  static constexpr const char* const ACTION_THINKING = "thinking";
  static constexpr const char* const ACTION_SHOUTING = "shouting";
  static constexpr const char* const ACTION_MURMURING = "murmuring";
  static constexpr const char* const FIRST_PERSON = "_1st";

  explicit Character(const std::string& dir, const std::string& dialogFormat, bool isFirstPerson = false); // non-display character
  Character(const std::string& dir, const std::string& dialogFormat, const std::string& poseFormat, const std::string& exprFormat, Pos2i bottomCenter, bool isFirstPerson = false, bool drawStand = true); // display character

private:
  void nextScene();
public:
  void keepsAllInNextScene();
  void changesExprInNextScene(const std::string& pose, const std::string& expression);
  void movesInNextScene(const std::string& pose, const std::string& expression, Pos2i moveTo);
  void speaksInNextScene(const std::string& pose, const std::string& expression, const drawable::TextLike& speaking, Action action = Action::NORMAL);

  std::shared_ptr<drawable::Texture> getStand();
  std::shared_ptr<drawable::Texture> getSpeakingDialog(std::shared_ptr<drawable::Texture>& speaking);

  void nextAct(bool isFirstPerson = false); // next act, clear FP flags

private:
  // status
  bool hasStandToDraw, drawStand;
  bool isFirstPerson;
  int eyeBinder;
  // unlikely to change ( changes only when first-person changes )
  std::shared_ptr<drawable::Texture> dialog, thinkingDialog, shoutingDialog, murmuringDialog;
  std::shared_ptr<drawable::Stand> stand{};
  // scene-wise
  Action action;
  Pos2i position;
  std::string poseFmt, exprFmt;
};

} }

#endif //DIALOGVIDEOGENERATOR_CHARACTER_H
