// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/drawable.h>

#include <cctype>
#include <random>
#include <common613/file_utils.h>
#include <dialog_video_generator/math/pos_arith.h>
#include <dialog_video_generator/math/random_utils.h>

using namespace common613;

namespace dialog_video_generator { namespace drawable {

Stand::Stand(const std::string& dir, const std::string& pose, const std::string& expression,
             unsigned int mul, Frames speaking, bool flip)
    : image{{}, mul, {}}, speaking(speaking), mouthStatus(0), eyeBlinkCountDown(nullptr), eyeStatus(0) {
  std::string posDir = dir + pose + '/', poseUpper(pose.size(), '\0');
  std::transform(pose.begin(), pose.end(), poseUpper.begin(), [](char c){return std::toupper(c);});
  image.raw.load(posDir, poseUpper, true);
  image.pos.x() = -checked_cast<Dim>(image.raw.size.w() * mul / 2);
  image.pos.y() = -checked_cast<Dim>(image.raw.size.h() * mul); // unchecked cast
  std::uint32_t pos[4];
  {
    File file = file::open(posDir + poseUpper + ".POS", "rb");
    file::read(file, pos, 4);
  }
  std::string expressionPrefix = poseUpper + "_" + expression;

  eye[0].raw.load(posDir, expressionPrefix + "_E0", true);
  eye[1].raw.load(posDir, expressionPrefix + "_E1", true);
  eye[2].raw.load(posDir, expressionPrefix + "_E2", true);
  eye[3].raw.load(posDir, expressionPrefix + "_E3", true);
  Vec2i eyeOffset = image.pos + Vec2i::of(flip ? eye[0].raw.size.w() - 1 - pos[0] : pos[0], pos[1]) * mul;
  for (auto& eyeImage : eye) {
    eyeImage.mul = mul;
    eyeImage.pos = eyeOffset;
    eyeImage.flipX = flip;
  }

  mouth[0].raw.load(posDir, expressionPrefix + "_M0", true);
  mouth[1].raw.load(posDir, expressionPrefix + "_M1", true);
  mouth[2].raw.load(posDir, expressionPrefix + "_M2", true);
  Vec2i mouthOffset = image.pos + Vec2i::of(flip ? mouth[0].raw.size.w() - 1 - pos[2] : pos[2], pos[3]) * mul;
  for (auto& mouthImage : mouth) {
    mouthImage.mul = mul;
    mouthImage.pos = mouthOffset;
    mouthImage.flipX = flip;
  }
}

Stand::~Stand() = default;

Frames Stand::duration() const {
  return speaking;
}

std::size_t Stand::bufferCount() const {
  return 3;
}

std::size_t Stand::nextFrame(Frames timeInScene) {
  int last = mouthStatus;
  int ret = 0;
  if (speaking > 0_fr && timeInScene <= speaking) {
    if (timeInScene.x() % 3 == 0) {
      switch (mouthStatus) {
      case 0: {
        std::bernoulli_distribution dist(0.8);
        mouthStatus = dist(gen) ? 2 : 1;
      }
        break;
      case 1: {
        std::bernoulli_distribution dist(0.8);
        mouthStatus = dist(gen) ? 2 : 0;
      }
        break;
      case 2: {
        std::bernoulli_distribution dist(0.8);
        mouthStatus = dist(gen) ? 1 : 0;
      }
        break;
      }
    }
  } else {
    mouthStatus = 0;
  }
  if (mouthStatus != last) {
    ret = 1;
  }
  if (eyeBlinkCountDown != nullptr) {
    if (eyeBlinkCountDown->x() < 7) {
      eyeStatus = eyeBlinkCountDown->x() >= 4 ? 6 - eyeBlinkCountDown->x() : eyeBlinkCountDown->x();
      ret = 2;
    }
    if (eyeBlinkCountDown->x() == 0) {
      refreshEyeBlinkCountDown(eyeBlinkCountDown);
    } else {
      --eyeBlinkCountDown->x();
    }
  }
  return ret;
}

void Stand::nextScene(bool stop, Frames point) {
  speaking = stop || point > speaking ? 0_fr : speaking - point;
}

void Stand::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  image.addTask(offset, true, alpha, false, tasks);
  eye[eyeStatus].addTask(offset, false, alpha, false, tasks);
  mouth[mouthStatus].addTask(offset, false, alpha, false, tasks);
}

void Stand::setSpeakingDuration(Frames duration) {
  speaking = duration;
}

void Stand::bindEyeStatus(Frames* countdown) {
  eyeBlinkCountDown = countdown;
}

void Stand::refreshEyeBlinkCountDown(Frames* countDown) {
  std::uniform_int_distribution<typename Frames::valueType> dist((3_sec).x(), (4_sec).x());
  countDown->x() = dist(gen);
}

} }
