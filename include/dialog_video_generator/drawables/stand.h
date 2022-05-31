// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#ifndef DIALOGVIDEOGENERATOR_STAND_H
#define DIALOGVIDEOGENERATOR_STAND_H

#include <string>
#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator {
namespace config {

extern std::string STAND_POSE_STRING_FORMAT;
extern std::string STAND_POSE_DIR_FORMAT;
extern std::string STAND_POSE_FILE_FORMAT;
extern std::string STAND_EXPR_EYE_FORMAT;
extern std::uint8_t STAND_EXPR_EYE_COUNT;
extern std::string STAND_EXPR_MOUTH_FORMAT;
extern std::uint8_t STAND_EXPR_MOUTH_COUNT;

}

namespace drawable {

class StillSelector : public StatusSelector {
public:
  explicit StillSelector(std::size_t st) : status(st) {}
  ~StillSelector() override = default;
  bool select(Frames) override { return false; }
private:
  std::size_t status;
};

class BlinkSelector : public StatusSelector {
public:
  explicit BlinkSelector(Frames* cd) : countDown(cd) {}
  ~BlinkSelector() override = default;
  bool select(Frames timeInShot) override;
  static void reset(BlinkSelector* instance);
private:
  void resetBlinkCountDownRandomly();
  Frames* countDown;
};

class SpeakingMouthSelector : public StatusSelector {
public:
  explicit SpeakingMouthSelector(Frames sp) : speaking(sp) {}
  ~SpeakingMouthSelector() override = default;
  bool select(Frames timeInShot) override;
  Frames leastDuration() const override { return speaking; }
  void setSpeakingFrames(const Frames& value) { this->speaking = value; }
  void nextShot(bool stop, Frames point) override;
private:
  Frames speaking;
};

inline StatusSelector* defaultMouthSelector(Frames speaking) {
  if (speaking) {
    return new SpeakingMouthSelector(speaking);
  } else {
    return new StillSelector(0);
  }
}

inline StatusSelector* defaultEyeSelector(Frames* countDown) {
  if (countDown) {
    return new BlinkSelector(countDown);
  } else {
    return new StillSelector(0);
  }
}

}
}

#endif //DIALOGVIDEOGENERATOR_STAND_H
