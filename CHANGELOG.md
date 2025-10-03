# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to **date based releases** from [PEP 440](https://peps.python.org/pep-0440/).
Scheme of releases: `v<4-digit year>.<1 or 2-digit month>.<patch starting at 0>`.

## [Unreleased]

### Added

- Added support LoRA.
- Added generation information in title.
- Added ADetailer support.
- Added megapixel resolution display in image size toolbar.
- Added support for loading models by dropping safetensors files.
- Added auto-find model feature that automatically searches for and loads the correct model when loading images with embedded metadata.
- Added system memory monitoring to status bar with real-time usage display.
- Added GPU memory usage monitoring to status bar with color-coded warnings and detailed tooltips.
- Added AI agent documentation for developers working with the codebase.

### Changed

- Now image changing is smooth.
- Change focus after push "generate" button to positive prompt editor.
- Enhanced prompt processing to support longer and more complex prompts without automatic truncation.
- Improved handling of advanced prompt syntax including better support for prompt combinations using "BREAK" keyword.
- Better reliability when using weighted prompts and complex prompt structures.

### Fixed

- Fixed rich text paste handling in prompt editors - formatting (bold/italic/colors/size) is now dropped during clipboard operations.
- Fixed CUDA cache leak where model data persisted after change model.
- Fixed potential file overwrite during automatic image saving.
- Fixed prompt processing issues with advanced syntax and improved reliability of prompt weighting features.

## [v2025.3.2] - 2025-03-18

### Added

- Added support v-prediction models.
- Added controls in the UI for v-prediction models.
- Added saving/loading generation metadata into/from XMP in JPEG files.
- Added loading metadata by dropping JPEG files into the application.

### Fixed

- Fixed error on generation after CUDA OutOfMemoryError. Clearing pipeline cache after OutOfMemoryError.

## [v2025.3.1] - 2025-03-13

### Added

- Added support for custom fonts, font weight, and font size for prompt editors.
- Added support for custom font weight for compel syntax highlighting in prompt editors.
- Added showing path of saved image after generating while auto-saving enabled.

### Fixed

- Fixed reading configuration file.
- Fixed autocompletion in prompts.

### Changed

- Changed place of configuration file.
- Improved documentation.

## [v2025.3.0] - 2025-03-06

### Added

- Using Stable Diffusion SDXL based models (SDXL, Pony, Illustrious, etc.) for rendering.
- UI to control rendering.
- Using [DeepCache](https://github.com/horseee/DeepCache) to dramatically accelerate rendering.
- Added support [Compel](https://github.com/damian0815/compel/blob/main/doc/syntax.md) for prompts.
