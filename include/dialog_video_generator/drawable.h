// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file drawable.h
/// @brief Types about @ref Drawable.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_DRAWABLE_H
#define DIALOGVIDEOGENERATOR_DRAWABLE_H

#include <dialog_video_generator/image.h>
#include <dialog_video_generator/math/time_utils.h>

namespace dialog_video_generator {
namespace drawable {

/**
 * @brief Base class for all drawable elements.
 *
 * Every drawable is a logical set of layers to paint onto a buffer.
 *
 * We only repaint drawables that are changed themselves, or are after a changed layer.
 * So @c bufferCount provides the number of layers, and @c nextFrame provides how many layers should be repainted.
 */
class Drawable : public std::enable_shared_from_this<Drawable> {
public:
  virtual ~Drawable() = default;

  /// @brief Get naturally minimal duration of the layer.
  COMMON613_NODISCARD virtual Frames duration() const = 0;

  /// @brief Gets the number of buffers needed to draw.
  /// @note It is also the number of layers.
  COMMON613_NODISCARD virtual std::size_t bufferCount() const = 0;

  /// @brief Calculates the status in the next frame, changes into it, and returns the number of layers to redraw.
  virtual std::size_t nextFrame(Frames timeInShot) = 0;

  /// @brief Stops the drawable and move it into another shot.
  /// @param stop Whether any animation should be stop if any.
  /// @param point The time point to continue in the next shot, if any animation is playing and not stopped.
  /// @return @c Drawable object to render in next shot.
  /// @retval shared_from_this if not changed.
  COMMON613_NODISCARD virtual std::shared_ptr<Drawable> nextShot(bool stop, Frames point) {
    return shared_from_this();
  }

  /// @brief Generates tasks to draw itself.
  /// @param offset The extra vector offset in the tasks.
  /// @param alpha The extra alpha in the tasks (0-16).
  /// @param[in,out] tasks Where generated tasks is appended.
  virtual void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const = 0;
};

/**
 * @brief Base class for all static, single-layer drawable elements.
 *
 * Static drawables are elements which need no repainting if not affected.
 */
class Static : public Drawable {
public:
  ~Static() override;

  /// Static drawables need no frames to animate.
  /// @retval 0_fr always.
  Frames duration() const final { return 0_fr; }

  /// Static drawables need only one layer.
  /// @retval 1 always.
  std::size_t bufferCount() const final { return 1; }

  /// Static drawables are always static.
  /// @retval 0 always.
  std::size_t nextFrame(Frames timeInShot) final { return 0; }

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;
};

/**
 * @brief Base class for all static image-based elements.
 */
class StaticImage : public Static {
public:
  friend class Drawable;

  /// @brief Import a image as the layer.
  explicit StaticImage(Image sprite);

  ~StaticImage() override;

  Size staticSize() const;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;

protected:
  Image image;
};

/**
 * @brief A non-transparent image.
 *
 * @note Alpha channels in the images loaded this way will be ignored.
 *
 * In earliest code, @c addTask does not have a external alpha channel.
 */
class EntireImage : public StaticImage {
public:
  /// @brief Loads a image from pathname @p dir and @p name (".png" will be appended)
  /// and specified multiplier @p mul and offset @p offset.
  EntireImage(const std::string& dir, const std::string& name, unsigned int mul = 1, Vec2i offset = {0, 0});

  /// @inheritdoc
  explicit EntireImage(Image sprite);

  ~EntireImage() override;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
};

/**
 * @brief A normal image texture.
 *
 * Render alpha channel.
 */
class Texture : public StaticImage {
public:
  /// @brief Loads a image from pathname @p dir and @p name (".png" will be appended)
  /// and specified multiplier @p mul and offset @p offset.
  Texture(const std::string& dir, const std::string& name, unsigned int mul = 1, Vec2i offset = {0, 0});

  /// @inheritdoc
  explicit Texture(Image sprite);

  ~Texture() override;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;
};

/**
 * @brief A colored rectangle to paint.
 */
class ColorRect : public Static {
public:
  /// @brief Initializes a screen-sized rectangle with a specified @p color.
  explicit ColorRect(Color4b color);

  /// @brief Initializes a color rectangle with a @p color, and position @p offset and size @p rect.
  ColorRect(Color4b color, Vec2i offset, Size rect);

  ~ColorRect() override;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

private:
  ColorImage colorImage;
};

/**
 * @brief Base class for frame-wise updating frames.
 *
 * Frame updating drawables are elements which need refreshing under a predictable procedure.
 */
class UpdatedByFrame : public Drawable {
public:
  Frames duration() const override = 0;
  std::size_t bufferCount() const override = 0;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;
};

/**
 * @brief Text boxes.
 *
 * Texts will be shown in order in a constant speed.
 * Note that the speed will be approximated as numbers of new characters for every frame.
 *
 * @warning TextLike does not provide frame-reentrant guarantee by now. It always changes to the next frame.
 */
class TextLike : public UpdatedByFrame {
public:
  /**
   * @brief Loads a string @p content, print it at @p pos, with width of @p sz.
   * @param colorType Whether it should be rendered in thinking color.
   * @param start The index of the character, before which characters are displayed before the first frame.
   * @param speedNum,speedDen In every new frame, @p speedNum / @p speedDen characters appears.
   * @param fontIndex Font index.
   */
  TextLike(const std::string& content, Vec2i pos, Size sz, bool colorType = false,
           std::size_t start = 0, std::size_t speedNum = 1, std::size_t speedDen = 2, std::size_t fontIndex = 0);

  ~TextLike() override;

  Frames duration() const override;
  std::size_t bufferCount() const override;
  /// @bug Texts shows more characters when called this, ignoring @p timeInShot !!!
  std::size_t nextFrame(Frames timeInShot) override;

  /// @warning Texts will keep the current status when crossing shots, which is convenient but not recommended.
  using Drawable::nextShot;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

private:
  std::vector<Image> glyphs;
  Size size;
  std::size_t start, current;
  std::size_t speedNum, speedDen;
  bool colorType; // 0 = speaking, black; 1 = thinking, blue
};

class StatusSelector {
public:
  virtual ~StatusSelector() = default;
  virtual bool select(Frames timeInShot) = 0;
  virtual Frames leastDuration() const { return 0_fr; }
  virtual void nextShot(bool stop, Frames point) {};
  std::size_t status{0};
};

/**
 * @brief Stand CG, or @e Tachie (<span lang="ja">立ち絵</span>).
 *
 * Provides functions for blinking and speaking stand CG update.
 * An eye status variable should be provided externally to bind here.
 *
 * @note The origin of a @p Stand is at the bottom-center position.
 *
 * @warning The eye and mouth states does not and will not provide frame-reentrant guarantee.
 */
class Stand : public UpdatedByFrame {
public:
  /**
   * @brief Loads stand CG resources from @p dir, @p pose and @p expression.
   *
   * @param dir Resource root dir.
   * @param character Character file string.
   * @param pose Character pose string.
   * @param expression Expression string.
   * @param mul Size multiplier.
   * @param speaking Duration for which the character is speaking.
   * @param flip Whether the CG should be flipped in X (horizontal) direction.
   *
   * @warning There is no user-defined path name pattern specification for the time being.
   * Maybe more flexible method will be added.
   */
  Stand(const std::string& dir,
        const std::string& character,
        const std::string& pose,
        const std::string& expression,
        int mul,
        Frames speaking = 0_fr,
        std::shared_ptr<StatusSelector> eyeCD = nullptr,
        bool flip = false);

  ~Stand() override;
  Frames duration() const override;
  std::size_t bufferCount() const override;
  /// @warning Calling this changes blink countdown.
  std::size_t nextFrame(Frames timeInShot) override;
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

  /// @brief Sets the stand will speak for @p frames.
  void setSpeakingDuration(Frames duration);

private:
  Image image; // body picture
  std::vector<Image> eye; // eye pictures
  std::vector<Image> mouth; // mouth pictures
  // for next frame
  std::shared_ptr<StatusSelector> eyeSelector; // eye status
  std::shared_ptr<StatusSelector> mouthSelector; // mouth status
};

/**
 * @brief A wrapper to translate a drawable.
 */
class Translated : public Drawable {
public:
  /// @brief Wraps @p target, and translate it by @p offset.
  Translated(std::shared_ptr<Drawable> target, Vec2i offset);

  ~Translated() override;

  Frames duration() const override { return target->duration(); }
  std::size_t bufferCount() const override { return target->bufferCount(); }
  std::size_t nextFrame(Frames timeInShot) override { return target->nextFrame(timeInShot); }
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override {
    target = target->nextShot(stop, point);
    return shared_from_this();
  }
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override;

private:
  std::shared_ptr<Drawable> target;
  Vec2i translatedOffset;
};

/**
 * @brief A convenient function to construct @ref Translated by @ref Translated#Translated(target, offset) .
 * @related Translated
 */
inline std::shared_ptr<Translated> translate(std::shared_ptr<Drawable> target, Vec2i offset) {
  return std::make_shared<Translated>(std::move(target), offset);
}

/**
 * @brief Base class for animated drawables.
 *
 * Animation states in each frame should be rendered in such a pipeline:
 * + calculated in @c nextFrame and stored in the object;
 * + used in @c addTask.
 */
class Animated : public Drawable {
public:
  /// @brief Sets the target drawable @p target to play an animation on.
  explicit Animated(std::shared_ptr<Drawable> target);

  ~Animated() override;
  Frames duration() const final;

  /// @brief Returns the least duration for this animation.
  /// @note Animation can be looped or infinite, a least animation duration should be provided.
  virtual Frames leastAnimationDuration() const = 0;

  std::size_t bufferCount() const final { return target->bufferCount(); }
  std::size_t nextFrame(Frames timeInShot) override = 0;
  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const override = 0;

protected:
  std::shared_ptr<Drawable> target;
};

/**
 * @brief Base class for animated drawables with moving-like animations.
 */
class Movement : public Animated {
public:
  /// @brief Sets the target drawable @p target and move duration @p duration .
  Movement(std::shared_ptr<Drawable> target, Frames duration);

  ~Movement() override;
  Frames leastAnimationDuration() const final;
  std::size_t nextFrame(Frames timeInShot) override;
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override;

  /// @brief Calculates offset in the specified frame in the moving process.
  virtual Vec2i calculateOffset(Frames timeInShot) const = 0;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const final;
protected:
  Frames dur;
  Vec2i frameOffset;
};

/**
 * @brief Base class for animated drawables with alpha-change animations.
 */
class AlphaChange : public Animated {
public:
  /// @brief Sets the target drawable @p target and move duration @p duration .
  AlphaChange(std::shared_ptr<Drawable> target, Frames duration);

  ~AlphaChange() override;
  Frames leastAnimationDuration() const final;
  std::size_t nextFrame(Frames timeInShot) override;
  std::shared_ptr<Drawable> nextShot(bool stop, Frames point) override;

  /// @brief Calculates alpha in the specified frame in the changing process.
  virtual int calculateAlpha(Frames timeInShot) const = 0;

  void addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const final;
protected:
  Frames dur;
  int frameAlpha;
};

}

using drawable::Drawable;

}

#endif //DIALOGVIDEOGENERATOR_DRAWABLE_H
