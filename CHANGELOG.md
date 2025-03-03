# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to **date based releases** from [PEP 440](https://peps.python.org/pep-0440/).
Scheme of releases: `<4-digit year>.<1 or 2-digit month>.<patch starting at 0>`. 

## [Unreleased]

### Added

- Using Stable Diffusion SDXL based models (SDXL, Pony, Illustrious, etc.) for rendering.
- UI to control rendering.
- Using [DeepCache](https://github.com/horseee/DeepCache) to dramatically accelerate rendering. 
With 50% of the steps cached and a cache interval of 3, the generation speed increased by ~30%.
