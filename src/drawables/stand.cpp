// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/drawable.h>
#include <dialog_video_generator/drawables/stand.h>

#include <cctype>
#include <random>
#include <common613/file_utils.h>
#include <dialog_video_generator/math/pos_arith.h>
#include <dialog_video_generator/math/random_utils.h>

using namespace common613;
using namespace fmt::literals;

namespace dialog_video_generator {
namespace config {

std::string STAND_POSE_STRING_FORMAT("talkchara{char}_{pose}");
std::string STAND_POSE_DIR_FORMAT("{dir}{pose_str}/");
std::string STAND_POSE_FILE_FORMAT("{pose_dir}{pose_str}.POS");
std::string STAND_EXPR_EYE_FORMAT("{pose_str}_{expr}_E{eye_index}");
std::uint8_t STAND_EXPR_EYE_COUNT = 4;
std::string STAND_EXPR_MOUTH_FORMAT("{pose_str}_{expr}_M{mouth_index}");
std::uint8_t STAND_EXPR_MOUTH_COUNT = 3;

}

namespace drawable {

Stand::Stand(const std::string& dir, const std::string& character, const std::string& pose, const std::string& expression,
             int mul, Frames speaking, std::shared_ptr<StatusSelector> eyeCD, bool flip)
    : image{{}, checked_cast<unsigned int>(mul), {}},
      eye(config::STAND_EXPR_EYE_COUNT), mouth(config::STAND_EXPR_MOUTH_COUNT),
      eyeSelector(std::move(eyeCD ? std::move(eyeCD) : std::shared_ptr<StatusSelector>(new StillSelector(0)))),
      mouthSelector(defaultMouthSelector(speaking)) {
  // assembling pose dir string
  std::string poseString = fmt::format(config::STAND_POSE_STRING_FORMAT, "char"_a = character, "pose"_a = pose);
  std::string poseDir = fmt::format(config::STAND_POSE_DIR_FORMAT, "dir"_a = dir, "pose_str"_a = poseString);
  std::string poseUpper(poseString.size(), '\0');
  std::transform(poseString.begin(), poseString.end(), poseUpper.begin(), [](char c) { return std::toupper(c); });
  // load pose body/face CG
  image.raw.load(poseDir, poseUpper, true);
  image.pos.x() = checked_cast<Dim>(image.raw.size.w() * mul / -2);
  image.pos.y() = checked_cast<Dim>(-image.raw.size.h() * mul); // unchecked cast
  image.flipX = flip;
  std::int32_t pos[4];
  {
    std::string poseFileName = fmt::format(config::STAND_POSE_FILE_FORMAT, "pose_dir"_a = poseDir, "pose_str"_a = poseUpper);
    File file = file::open(poseFileName, "rb");
    file::read(file, pos, 4);
  }
  // load eye CG
  for (int i = 0; i < config::STAND_EXPR_EYE_COUNT; ++i) {
    eye[i].raw.load(
        poseDir,
        fmt::format(config::STAND_EXPR_EYE_FORMAT,
                    "pose_str"_a = poseUpper,
                    "expr"_a = expression,
                    "eye_index"_a = i),
        true);
  }
  Vec2i eyeOffset =
      image.pos + Vec2i::of(flip ? image.raw.size.w() - eye[0].raw.size.w() - pos[0] : pos[0], pos[1]) * mul;
  for (auto& eyeImage: eye) {
    eyeImage.mul = mul;
    eyeImage.pos = eyeOffset;
    eyeImage.flipX = flip;
  }
  // load mouth CG
  for (int i = 0; i < config::STAND_EXPR_MOUTH_COUNT; ++i) {
    mouth[i].raw.load(
        poseDir,
        fmt::format(config::STAND_EXPR_MOUTH_FORMAT,
                    "pose_str"_a = poseUpper,
                    "expr"_a = expression,
                    "mouth_index"_a = i),
        true);
  }
  Vec2i mouthOffset = image.pos + Vec2i::of(flip ? image.raw.size.w() - mouth[0].raw.size.w() - pos[2] : pos[2],
                                            pos[3]) * mul;
  for (auto& mouthImage: mouth) {
    mouthImage.mul = mul;
    mouthImage.pos = mouthOffset;
    mouthImage.flipX = flip;
  }
}

Stand::~Stand() = default;

Frames Stand::duration() const {
  return mouthSelector->leastDuration();
}

std::size_t Stand::bufferCount() const {
  return 3;
}

std::size_t Stand::nextFrame(Frames timeInShot) {
  int ret = 0;
  if (mouthSelector->select(timeInShot)) {
    ret = 1;
  }
  if (eyeSelector->select(timeInShot)) {
    ret = 2;
  }
  return ret;
}

std::shared_ptr<Drawable> Stand::nextShot(bool stop, Frames point) {
  mouthSelector->nextShot(stop, point);
  eyeSelector->nextShot(stop, point);
  return shared_from_this();
}

void Stand::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  image.addTask(offset, true, alpha, false, tasks);
  eye[eyeSelector->status].addTask(offset, false, alpha, false, tasks);
  mouth[mouthSelector->status].addTask(offset, false, alpha, false, tasks);
}

void Stand::setSpeakingDuration(Frames duration) {
  if (auto selector = dynamic_cast<SpeakingMouthSelector*>(mouthSelector.get())) {
    selector->setSpeakingFrames(duration);
  }
}

bool BlinkSelector::select(Frames timeInShot) {
  static std::size_t changeStart = 2 * config::STAND_EXPR_EYE_COUNT - 1;
  auto cd = countDown->x();
  if (cd < changeStart) {
    status = cd > config::STAND_EXPR_EYE_COUNT ? changeStart - 1 - cd : cd;
    resetBlinkCountDownRandomly();
    return true;
  }
  return false;
}

void BlinkSelector::resetBlinkCountDownRandomly() {
  if (*countDown) {
    --countDown->x();
  } else {
    std::uniform_int_distribution<typename Frames::valueType> dist((2_sec).x(), (5_sec).x());
    countDown->x() = dist(random::gen);
  }
}

void BlinkSelector::reset(BlinkSelector* instance) {
  instance->resetBlinkCountDownRandomly();
}

bool SpeakingMouthSelector::select(Frames timeInShot) {
  if (speaking > 0_fr && timeInShot <= speaking) {
    if (timeInShot.x() % 3 == 0) {
      switch (status) {
        case 0: {
          std::bernoulli_distribution dist(0.8);
          status = dist(random::gen) ? 2 : 1;
        }
          break;
        case 1: {
          std::bernoulli_distribution dist(0.8);
          status = dist(random::gen) ? 2 : 0;
        }
          break;
        case 2: {
          std::bernoulli_distribution dist(0.8);
          status = dist(random::gen) ? 1 : 0;
        }
          break;
      }
      return true;
    }
  } else {
    if (status != 0) {
      status = 0;
      return true;
    }
    return false;
  }
}

void SpeakingMouthSelector::nextShot(bool stop, Frames point) {
  speaking = stop || point > speaking ? 0_fr : speaking - point;
}

} }
