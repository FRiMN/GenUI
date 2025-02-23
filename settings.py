from typing import Type, Tuple

from pydantic import DirectoryPath, BaseModel
from pydantic.types import NewPath
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource


class BaseGenUISettings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file="config.toml",
        env_prefix="genui_",
        cli_parse_args=True,
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


class AutoSaveImageSettings(BaseModel):
    enabled: bool = False
    path: DirectoryPath | NewPath = "./result/"


class Settings(BaseGenUISettings):
    autosave_image: AutoSaveImageSettings = AutoSaveImageSettings()


settings = Settings()
