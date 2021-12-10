// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_DRAWABLE_H
#define DIALOGVIDEOGENERATOR_DRAWABLE_H

#include <dialog_video_generator/image.h>
#include <dialog_video_generator/math/time_utils.h>

namespace dialog_video_generator { namespace drawable {

class Drawable {
public:
  virtual ~Drawable() = default;
  virtual Frames duration() const = 0;
  virtual std::size_t bufferCount() const = 0;
  virtual std::size_t nextFrame(Frames timeInScene) = 0;
  virtual void nextScene(bool stop, Frames point) = 0;
  virtual void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const = 0;
};

class Static : public Drawable {
public:
  friend class Drawable;
  explicit Static(Image sprite);
  ~Static() override;
  Frames duration() const final { return 0_fr; }
  std::size_t bufferCount() const final { return 1; }
  std::size_t nextFrame(Frames timeInScene) final { return 0; }
  void nextScene(bool stop, Frames point) final {}
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;
protected:
  Image image;
};

class EntireImage : public Static {
public:
  EntireImage(const std::string& dir, const std::string& name, unsigned int mul = 1, Vec2i offset = {0, 0});
  explicit EntireImage(Image sprite);
  ~EntireImage() override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
};

class Texture : public Static {
public:
  Texture(const std::string& dir, const std::string& name, unsigned int mul = 1, Vec2i offset = {0, 0});
  explicit Texture(Image sprite);
  ~Texture() override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
};

class UpdatedByFrame : public Drawable {
public:
  Frames duration() const override = 0;
  std::size_t bufferCount() const override = 0;
  void nextScene(bool stop, Frames point) override = 0;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;
};

class TextLike : public UpdatedByFrame {
public:
  TextLike(const std::string& content, Vec2i pos, Size sz, bool colorType = false,
           std::size_t start = 0, std::size_t speedNum = 1, std::size_t speedDen = 2);
  ~TextLike() override;
  Frames duration() const override;
  std::size_t bufferCount() const override;
  std::size_t nextFrame(Frames timeInScene) override;
  void nextScene(bool stop, Frames point) final {}
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
private:
  std::vector<Image> glyphs;
  Size size;
  std::size_t start, current;
  std::size_t speedNum, speedDen;
  bool colorType; // 0 = speaking, black; 1 = thinking, blue
};

class Stand : public UpdatedByFrame {
public:
  Stand(const std::string& dir, const std::string& pose, const std::string& expression,
        unsigned int mul, Frames speaking = 0_fr, bool flip = false);
  ~Stand() override;
  Frames duration() const override;
  std::size_t bufferCount() const override;
  std::size_t nextFrame(Frames timeInScene) override;
  void nextScene(bool stop, Frames point) override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
  void setSpeakingDuration(Frames duration);
  void bindEyeStatus(Frames* countDown);
  static void refreshEyeBlinkCountDown(Frames* countDown);
private:
  Image image;
  Image eye[4];
  Image mouth[3];
  Frames speaking;
  // for next frame
  Frames* eyeBlinkCountDown; // (external)
  std::size_t eyeStatus; // eye status
  std::size_t mouthStatus; // mouth status
};

class Translated : public Drawable {
public:
  Translated(std::shared_ptr<Drawable> target, Vec2i offset);
  ~Translated() override;
  Frames duration() const override {
    return target->duration();
  }
  std::size_t bufferCount() const override {
    return target->bufferCount();
  }
  std::size_t nextFrame(Frames timeInScene) override {
    return target->nextFrame(timeInScene);
  }
  void nextScene(bool stop, Frames point) override {
    target->nextScene(stop, point);
  }
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
private:
  std::shared_ptr<Drawable> target;
  Vec2i translatedOffset;
};

inline std::shared_ptr<Translated> translate(std::shared_ptr<Drawable> target, Vec2i offset) {
  return std::make_shared<Translated>(std::move(target), offset);
}

class Animated : public Drawable {
public:
  explicit Animated(std::shared_ptr<Drawable> target);
  ~Animated() override;
  Frames duration() const final;
  virtual Frames leastAnimationDuration() const = 0;
  std::size_t bufferCount() const final {
    return target->bufferCount();
  }
  std::size_t nextFrame(Frames timeInScene) override = 0;
  void nextScene(bool stop, Frames point) override = 0;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;
protected:
  std::shared_ptr<Drawable> target;
};

class Movement : public Animated {
public:
  Movement(std::shared_ptr<Drawable> target, Frames duration);
  ~Movement() override;
  Frames leastAnimationDuration() const final;
  std::size_t nextFrame(Frames timeInScene) override;
  void nextScene(bool stop, Frames point) override;
  virtual Vec2i calculateOffset(Frames timeInScene) const = 0;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const final;
protected:
  Frames dur;
  Vec2i frameOffset;
};

class Shouting : public Movement {
public:
  explicit Shouting(std::shared_ptr<Drawable> target);
  ~Shouting() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  constexpr static const long double LENGTH_SECOND = 0.5;
};

inline std::shared_ptr<Shouting> animateShouting(std::shared_ptr<Drawable> target) {
  return std::make_shared<Shouting>(std::move(target));
}

class Murmuring : public Movement {
public:
  explicit Murmuring(std::shared_ptr<Drawable> target);
  ~Murmuring() override;
  Vec2i calculateOffset(Frames timeInScene) const override;
private:
  constexpr static const long double LENGTH_SECOND = 2;
};

inline std::shared_ptr<Shouting> animateMurmuring(std::shared_ptr<Drawable> target) {
  return std::make_shared<Shouting>(std::move(target));
}

}

using drawable::Drawable;

}

#endif //DIALOGVIDEOGENERATOR_DRAWABLE_H
