from pathlib import Path

from pydantic import BaseModel
from pydantic.types import NewPath, StrictInt, FilePath, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource
from platformdirs import user_config_dir
from PyQt6.QtGui import QFont


APP_NAME = "genui"

CONFIG_FILE_PATH = Path(user_config_dir(APP_NAME)) / "config.toml"
print(f"{CONFIG_FILE_PATH=}")
if CONFIG_FILE_PATH.exists():
    print("User config exists")
else:
    print("User config does not exist")


class BaseGenUISettings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=CONFIG_FILE_PATH,
        env_prefix="genui_",
        cli_parse_args=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


class AutoSaveImageSettings(BaseModel):
    enabled: bool = False
    path: DirectoryPath | NewPath = Path("./result/")


class DeepCacheSettings(BaseModel):
    cache_interval: StrictInt = 3
    cache_branch_id: StrictInt = 0
    skip_mode: str = "uniform"
    
    
class PromptEditorSettings(BaseModel):
    font_family: str | None = None
    font_size: int = 10
    font_weight: int = QFont.Weight.Normal
    compel_font_weight: int = QFont.Weight.Bold
    
    
class ADetailerSettings(BaseModel):
    yolov_model_path: FilePath | None = None
    prompt: str = ""
    n_prompt: str = ""
    inpaint_strength: float = 0.4
    target_size: tuple[int, int] = (512, 512)
    inference_steps: int = 20
    mask_dilation: int = 4
    mask_blur: int = 4
    mask_padding: int = 32
    

class Settings(BaseGenUISettings):
    autosave_image: AutoSaveImageSettings = AutoSaveImageSettings()
    deep_cache: DeepCacheSettings = DeepCacheSettings()
    prompt_editor: PromptEditorSettings = PromptEditorSettings()


settings = Settings()
print(f"SETTINGS={settings.json()}")
