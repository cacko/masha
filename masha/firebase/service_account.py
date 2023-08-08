from pathlib import Path
from firebase_admin import credentials, App, initialize_app, db
from typing import Optional
from masha.config import app_config


class ServiceAccountMeta(type):
    _instance: Optional["ServiceAccount"] = None
    _admin_json: Optional[Path] = None

    SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/firebase.database",
    ]

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = type.__call__(cls, *args, **kwargs)
        return cls._instance

    def register(cls, service_account_file: Path):
        cls._admin_json = service_account_file

    @property
    def admin_json(cls) -> Path:
        assert cls._admin_json
        return cls._admin_json

    @property
    def app(cls) -> App:
        return cls().get_app()


class ServiceAccount(object, metaclass=ServiceAccountMeta):

    __credentials: Optional[credentials.Certificate] = None
    __app: Optional[App] = None

    def get_app(self) -> App:
        if not self.__app:
            self.__app = initialize_app(
                self.get_credentials(), dict(databaseURL=app_config.firebase.rdb_host)
            )
        return self.__app

    def get_credentials(self) -> credentials.Certificate:
        if not self.__credentials:
            self.__credentials = credentials.Certificate(
                self.__class__.admin_json.as_posix()
            )
        return self.__credentials

ServiceAccount.register(Path(app_config.firebase.service_json))
app = ServiceAccount.app