// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Abstraction for characters.

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
/// @brief Magnification for stands.
extern std::uint16_t STAND_MULTIPLIER;
}

namespace abstraction {

/// @brief Types of actions.
enum class Action {
  NO_CHANGE = -1,
  NORMAL = 0,
  THINKING,
  SHOUTING,
  MURMURING,
};

/**
 * @brief Holder class for character information and status. Encapsulates stand changing process for characters.
 *
 * @warning Use @c CharacterToDraw if you are to use it like a @c Drawable .
 */
class Character {
public:
  static constexpr const char* const ACT_NORMAL = "";
  static constexpr const char* const ACT_THINKING = "_th";
  static constexpr const char* const ACT_SHOUTING = "_sh";
  static constexpr const char* const ACT_MURMURING = "_mur";
  static constexpr const char* const FIRST_PERSON = "_1st";

  /// @brief Constructor for non-display character.
  Character(const std::string& dialogDir, const std::string& dialogFormat, bool firstPerson = false);
  /// @brief Constructor for display character. Use CharacterToDraw instead if you want to use it as a @c Drawable.
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
  /// @brief Sets offset to @p newOffset for stands created later.
  void setOffset(Vec2i newOffset);

  /// @brief Sets the character to leave expression and action unchanged, keep silent in the next shot.
  void keepsAllInNextShot();
  /// @brief Sets the character to change to the given expression, but keep action unchanged, keep silent in the next shot.
  void changesExprInNextShot(const std::string& pose, const std::string& expression, bool flip = false);
  // @todo not completed. Maybe moved outside later.
  void movesInNextShot(const std::string& pose, const std::string& expression, Vec2i newOffset);
  /// @brief Sets the character to speak and change action, but keep expression in the next shot.
  void speaksInNextShot(const std::shared_ptr<drawable::TextLike>& lines, Action newAction = Action::NORMAL,
                        bool addMouthAnimation = true);
  /// @brief Sets the character to speak and change action and expression in the next shot.
  void speaksAndChangesExprInNextShot(const std::shared_ptr<drawable::TextLike>& lines,
                                      const std::string& pose, const std::string& expression, bool flip = false,
                                      Action newAction = Action::NORMAL, bool addMouthAnimation = true);

  /// @brief Construct a stand using currently set expression, action and mouth movement duration.
  /// @return a @c Drawable , usually a @c Stand , maybe wrapped with a @c Movement .
  std::shared_ptr<drawable::Drawable> getStand();
  /// @brief Construct a text box with a dialog background, using currently set lines and action.
  /// @return a @c Drawable , usually a @c Dialog , maybe wrapped with a @c Movement .
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
  // shot-wise
  Action action;
  bool actionAnimated; // true: depress action animation. useful when the shot follows another which has animated it.
  Vec2i offset;
  std::string poseFmt, exprFmt;
};

} }

#endif //DIALOGVIDEOGENERATOR_CHARACTER_H
