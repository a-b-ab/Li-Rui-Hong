"""
    Created by crow at 2024-12-05.
    Description: 项目整体配置文件
    Changelog: all notable changes to this file will be documented
"""

import os

from src.utils.tools import read_file


class Config:
    """
    项目配置
    """

    # 应用配置
    DEBUG = True
    TiMEZONE = "Asia/Shanghai"
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    PROJECT_NAME = os.getenv("PROJECT_NAME", os.path.basename(ROOT_DIR))
    API_DIR = os.path.join(BASE_DIR, "views")
    HOST = os.getenv("HOST", "0.0.0.0")
    HTTP_PORT = os.getenv("HTTP_PORT", 8765)
    WORKERS = os.getenv("WORKERS", 1)

    APP_ID_CONFIG = {"env_city_api": os.getenv("ENV_CITY_TOKEN", "12138")}

    MONGODB_CONFIG = {
        "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        "operate_db": os.getenv("MONGODB_DB", "env_city_test"),
    }

    # 日志配置
    TAG = {
        "info": f"{PROJECT_NAME.replace('_','-')}-info",
        "warn": f"{PROJECT_NAME.replace('_','-')}-warn",
        "error": f"{PROJECT_NAME.replace('_','-')}-error",
    }

    @staticmethod
    def get_version() -> str:
        """获取当前服务版本, 需要自定义 version 文件"""
        version_list = read_file(os.path.join(Config.ROOT_DIR, "version"))
        return version_list[0] if version_list else "undefined"


if __name__ == "__main__":
    print(Config.API_DIR)
