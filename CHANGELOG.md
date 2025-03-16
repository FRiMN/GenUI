# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to **date based releases** from [PEP 440](https://peps.python.org/pep-0440/).
Scheme of releases: `v<4-digit year>.<1 or 2-digit month>.<patch starting at 0>`. 

## [Unreleased]

### Added

- Added support v-prediction models.
- Added controls in the UI for v-prediction models.
- Added saving/loading generation metadata into/from XMP in JPEG files.
- Added loading metadata by dropping JPEG files into the application.

### Fixed

### Changed

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
