# NINGYOV

Welcome to Ningyov (pronounced as ning-GHYAW),
the library to animate standing drawings into VIDEOs in the style of galgames or visual novels.

> ~~The name Ningyov is short for “Ningyov Is Not for Generating Your Own Videos.”~~
> The name Ningyov is derived from the Japanese word “<span lang="ja">人形</span>”(<span lang="ja">にんぎょう</span>, *ningyō*, lit. a doll or a puppet),
> but spelled with a V (instead of normally a circumflex, a macron, a U or an H)
> to avoid name collision.

Note that we generate video clips instead of games.

## Why you may need this library?

### You need this library if you:

+ Want to generate video galgame/visual video-style interactive videos with *tachie*s (standing drawings of characters) and dialogs.
+ Have a brilliant gameplay idea to write but do not excel at drawing to animate it.
+ Have a strong wish to exhibit your play with a video of an exact solution and/or frame rate avoiding resampling.

### You may use the following to replace this:

+ A galgame engine and a screen recorder.
+ An employed trained painter. ( A large amount of money required. )
+ Boring, tiresome and repetitive effort to arrange the play in video editors. ( Plenty of spare time required. )

### Compared with game engine + screen recorder:

There is a lot of galgame / visual novel engines on GitHub.
However, when it comes to game-like videos, there is somehow an absence.

#### When you should use Ningyov:

+ It is a WASTE of the resolution and frame rate of your materials and video clips.
  + Screen recording is limited to the frame rate of the area selected.
    + You will get interpolated results, which may be blurred.
  + Screen recording requires CPU calculation and may affect the frame rate of the game.
    + The selected frame rate of your video may not be well leveraged.

#### When you should use game engine + screen recorder:

+ Rapid rendering is preferred. Ningyov sacrifices some rendering time to realize video-level resolutions and frame rates.
+ Scenes are too complicated for Ningyov to describe. 
  + For example, Ningyov is for 2D animations, Live-2D or 3D scenes are not supported.
  + Some movements are difficult to describe in the frame-based way.
+ Some materials are videos itself.
  + Ningyov supports only images for now.

#### When you should use video editor only:

+ You do not know how to program and you have enough time to do repetitive works.

### What this library will do for you:

+ Generates video clips from your code.

### What else you should do to make a video:

A video editor is still required for after-effects.
The following are what you may want to do, and they are what we really SUGGEST you to do:

+ Edit scene transitions and other special effects.
+ Connect groups of clips into long videos.
+ Add soundtracks, for both background musics and sound effects to resemble a real game.
+ Arrange the clips on video platforms to add interactions. Of course ;)

## Documentation

### Dependency

#### Device Requirement

+ A compiler supporting `C++14`.
+ A GPU supporting `CUDA`. We need it to render frames.
  (The library was written for private use first, so it was designed for my own hardware.
  In this library, GPU renders the frames, while CPU encodes the video.)

We have tested the library on the following environments:
+ OS: Windows 8.1 / CUDA: 10.2 / Compiler: Visual Studio 2017
+ OS: Ubuntu 16.04 / CUDA: 10.2 / Compiler: gcc 5.4.0

#### Build Dependency

+ CUDA. We use 10. the version should not matter.
+ `Common613`. <https://github.com/613-forever/Common613>
+ `png++`. <https://www.nongnu.org/pngpp/>
+ `tiny-utf8`. <https://github.com/DuffsDevice/tiny-utf8>
+ `fmtlib`. We suggest installing via your package manager. <https://github.com/fmtlib/fmt>
+ `freetype`. We suggest installing via your package manager. <https://freetype.org>
+ `Boost.stacktrace`, `Boost.log`, `Boost.process`. We use `1.66` but it won't matter if compatible.

#### Run Dependency

+ `FFmpeg`. We suggest installing via your package manager. We use it as an external tool to encode videos, 
  so path variables should be prepared to find the executables.

### Library Structure

The library is composed of 4 abstraction levels.

+ Low-level: byte-level rendering objects. (CUDA programming files, `Image`/`RawImage`, et al.)
+ Drawable: drawable objects and paint engines. (`Texture`, `Stand`, `Engine`, et al.)
+ Semantic: common controlling objects in games. (`Characters`, `Dialog`, et al.)
+ Abstract: a director directing the action of objects. (`Director`, this is not committed as API is not concluded.)

### API Reference

You can generate documents with doxygen for the project, building the target `doc` yourself.
A compiled updating version may also be provided with GitHub Pages later.

### Examples

We may provide demonstration and examples in another repo later.
