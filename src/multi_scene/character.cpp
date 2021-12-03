// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/multi_scene/character.h>

#include <dialog_video_generator/drawable.h>

using namespace dialog_video_generator::drawable;

namespace dialog_video_generator {
namespace config {
std::uint16_t STAND_MULTIPLIER = 3;
}

namespace multi_scene {

Character::Character(const std::string& dir, const std::string& dialogFormat, bool isFirstPerson /* = false */) :
    hasStandToDraw(false), drawStand(false), isFirstPerson(isFirstPerson), eyeBinder{}, action{}, position{},
    poseFmt{}, exprFmt{} {
  if (isFirstPerson) {
    dialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_NORMAL));
    thinkingDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_THINKING));
    shoutingDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_SHOUTING));
    murmuringDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_MURMURING));
  } else {
    dialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_NORMAL));
    thinkingDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_THINKING));
    shoutingDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_SHOUTING));
    murmuringDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_MURMURING));
  }
}

Character::Character(const std::string& dir, const std::string& dialogFormat,
                     const std::string& poseFormat, const std::string& exprFormat, Pos2i bottomCenter,
                     bool isFirstPerson /* = isFalse */, bool drawStand /* = true */) :
    hasStandToDraw(true), drawStand(drawStand), isFirstPerson(isFirstPerson), eyeBinder(0), action{Action::NORMAL}, position{},
    poseFmt(poseFormat), exprFmt(exprFormat) {
  if (isFirstPerson) {
    dialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_NORMAL));
    thinkingDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_THINKING));
    shoutingDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_SHOUTING));
    murmuringDialog =
        std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat + FIRST_PERSON, ACTION_MURMURING));
  } else {
    dialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_NORMAL));
    thinkingDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_THINKING));
    shoutingDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_SHOUTING));
    murmuringDialog = std::make_shared<drawable::Texture>(dir, fmt::format(dialogFormat, ACTION_MURMURING));
  }
}

void Character::nextScene() {}

}
}
