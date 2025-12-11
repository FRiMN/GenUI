# Configuration Documentation

## Overview
The Genui application is designed to provide a set of customizable settings that allow users to tailor the application's behavior to their preferences. These settings are managed through a configuration file named `config.toml`, which is located in the user's configuration directory specific to the application.

About syntax of TOML file you can find more information in [official documentation](https://toml.io/en/).

## Configuration File Path
By default, the path to the configuration file (`config.toml`) for Genui on Linux is:
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

### 4. Auto-Find Model
This section allows you to configure automatic model discovery when loading images with embedded metadata. When enabled, the application will search for models referenced in image metadata within a specified directory.

- **enabled**: A boolean value that determines whether auto-find model is enabled (`False` by default).
- **path**: The directory where the application will search for models recursively. This should be the root directory containing your model files. The path may be relative to the current directory or an absolute path. This defaults to `None` (not set).

When you load an image that contains model information in its metadata, and the current model doesn't match the one used to generate the image, this feature will automatically search for and load the correct model if found in the specified path.

Example:
```toml
[autofind_model]
enabled = true
path = "/path/to/models/directory"
```

### 5. Auto-Find LoRAs
This section allows you to configure automatic discovery of LoRAs when loading images with embedded metadata. When enabled, the application will recursively scan the specified directory and populate the LoRA table in the UI automatically, for LoRAs referenced in image metadata within a specified directory.

- **enabled**: A boolean value that determines whether auto-find LoRAs is enabled (`False` by default).
- **path**: The directory where the application will search for LoRA files (usually `.safetensors`). The path may be relative to the current directory or an absolute path. This defaults to `None` (not set).

Example:
```toml
[autofind_loras]
enabled = true
path = "/path/to/loras/directory"
```

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

[autofind_model]
enabled = false
path = None

[autofind_loras]
enabled = true
path = "/home/user/models/loras"
```
