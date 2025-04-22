"""
Created by crow at 2024-12-05.
Description: 项目HTTP启动文件
    - 启动: pipenv run python ./src/http_app.py
Changelog: all notable changes to this file will be documented
"""

import asyncio
import threading

import requests

from flask import Flask
from flask_cors import CORS

from collector.env_huizhou_async import run_spider
from config import LOGGER, Config
from views.bp_api import bp_api


def create_app():
    """
    建立web应用
    """
    flask_app = Flask(__name__)
    CORS(flask_app)

    with flask_app.app_context():
        from databases import MongodbBase, MongodbManager

        # 初始化 MongoDB
        mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
            mongodb_config=Config.MONGODB_CONFIG
        )

        # 项目内部配置
        flask_app.config["app_config"] = Config
        flask_app.config["mongodb_base"] = mongodb_base
        flask_app.config["app_logger"] = LOGGER
        flask_app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

        # 全局请求 session
        req_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200)
        req_session.mount("http://", adapter)
        req_session.mount("https://", adapter)
        flask_app.config["req_session"] = req_session
        flask_app.config["app_logger"] = LOGGER

        # 打印启动日志
        api_version = Config.get_version()
        LOGGER.info(
            f"Service({Config.PROJECT_NAME} started successfully:{api_version})"
        )

        # 启动定时任务
        def run_spider_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_spider())

        # 在新线程中启动定时任务
        spider_thread = threading.Thread(target=run_spider_task)
        spider_thread.daemon = (
            True  # 设置为守护线程，这样主程序退出时，这个线程也会退出
        )
        spider_thread.start()

    flask_app.register_blueprint(bp_api)
    return flask_app


app = create_app()

if __name__ == "__main__":
    app.run(port=Config.HTTP_PORT, debug=Config.DEBUG)
