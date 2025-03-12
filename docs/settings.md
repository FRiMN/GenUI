# Configuration Documentation

## Overview
The Genui application is designed to provide a set of customizable settings that allow users to tailor the application's behavior to their preferences. These settings are managed through a configuration file named `config.toml`, which is located in the user's configuration directory specific to the application.

About syntax of TOML file you can find more information in [official documentation](https://toml.io/en/).

## Configuration File Path
By default, the path to the configuration file (`config.toml`) for Genui is:
```
/home/user/.config/genui/config.toml
```
This path is determined based on your operating system and user profile. If the file does not exist, you can create it manually. The app does not require a configuration file to work correctly. 

## Available Settings
The following settings can be configured in the `config.toml` file:

### 1. Image auto-save
This section allows you to enable or disable automatic saving of image results and specify a directory for these saved images.

- **enabled**: A boolean value that determines whether auto-saving is enabled (`False` by default).
- **path**: The directory where the saved images will be stored. The path may be relative to the current directory or an absolute path. This defaults to `./result/`. If directory does not exist, it will be created automatically.

Example:
```toml
[autosave_image]
enabled = true
path = "/path/to/save/directory"
```

### 2. DeepCache
This section allows you to configure the caching mechanism provided by [DeepCache](https://github.com/horseee/DeepCache) and using by the application.
More information about DeepCache parameters can be found in the [HuggingFace Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/optimization/deepcache).

- **cache_interval**: `3` by default.
- **cache_branch_id**: `0` by default.
- **skip_mode**: This defaults to `uniform`.

Example:
```toml
[deep_cache]
cache_interval = 6
cache_branch_id = 1
skip_mode = "specific"
```

### 3. Prompt Editor
This section allows you to configure the font settings for the prompt editor (positive and negative).

- **font_family**: The font family used in the prompt editor. This defaults to `None` and use system default font.
- **font_size**: The font size (integer) used in the prompt editor. This defaults to `10`.
- **font_weight**: The font weight (integer) used in the prompt editor. Can be `100`, `200`, `300`, `400`, `500`, `600`, `700`, `800`, `900`. This defaults to `400`.

Example:
```toml
[prompt_editor]
font_family = "Arial"
font_size = 10
font_weight = 200
```

## Configuration File Location
The location of the configuration file can be overridden by setting the environment variable `GENUI_CONFIG_FILE` before running the application. If this variable is set, it will point to a different path for the configuration file.

## Environment Variables
To modify settings through environment variables, prefix the parameter names with `genui_`. For example:
- To change the auto-save image setting, you can use an environment variable like `GENUI_AUTOSAVE_IMAGE_ENABLED`.

## Command Line Interface (CLI)
Genui supports command line interface options to modify settings. For see allowed options, run `genui --help`. Settings setted by CLI will override settings setted by configuration file.

## Example Configuration File
Here is an example of how the configuration file (`config.toml`) might look:
```toml
[autosave_image]
enabled = false
path = "./result/"

[deep_cache]
cache_interval = 3
cache_branch_id = 0
skip_mode = "uniform"

[prompt_editor]
font_family = None
font_size = 10
font_weight = 400
```
