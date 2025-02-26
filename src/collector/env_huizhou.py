"""
    Created by crow at 2025-02-26.
    Description: 获取惠州的空气质量和大气压数据(24小时)
    Changelog: all notable changes to this file will be documented
"""

import json
import re

from datetime import datetime

import requests

from lxml import html

from src.config import LOGGER, Config
from src.databases import (
    MongodbBase,
    MongodbManager,
    mongodb_find_by_page,
    mongodb_insert_many_data,
)

# Splash 服务的 URL
splash_url = "http://0.0.0.0:8050/render.html"


def fetch_and_process_data():
    """获取并处理数据"""
    # 请求参数
    params = {
        "url": "https://datashareclub.com/weather/%E5%B9%BF%E4%B8%9C/%E6%83%A0%E5%B7%9E/101280301.html",  # 目标网页
        "wait": 2,  # 等待页面加载的时间（秒）
    }

    # 发送请求给 Splash
    response = requests.get(splash_url, params=params, timeout=20)

    data_list = []

    # 获取页面内容
    if response.status_code == 200:
        page_content = response.text  # 获取 HTML 内容
        if page_content:
            tree = html.fromstring(page_content)
            # 使用 XPath 提取包含js变量的script标签
            data_content = tree.xpath(
                '//script[contains(text(),"var aqi_data")]/text()'
            )
            aqi_data = (
                re.findall(r"var aqi_data = (\[.*?\]);", data_content[0], re.DOTALL)
                if data_content
                else []
            )
            qy_data = (
                re.findall(r"var qy_data = (\[.*?\]);", data_content[0], re.DOTALL)
                if data_content
                else []
            )

            # 数据清洗
            aqi_data = json.loads(aqi_data[0].replace(",]", "]"))
            qy_data = json.loads(qy_data[0].replace(",]", "]"))

            # 数据提取
            data_list = [
                {
                    "datetime": int(
                        datetime.strptime(aqi[0], "%Y-%m-%d %H:%M").timestamp()
                    ),
                    "aqi": aqi[1],
                    "hap": qy[1],
                    "area": "惠州",
                }
                for aqi, qy in zip(aqi_data, qy_data)
            ]
        else:
            print("页面内容为空")
    else:
        print("请求失败:", response.status_code)
    return data_list


def env_data2mongodb(data: list):
    """将城市空气质量数据存入mongodb"""
    try:
        mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
            mongodb_config=Config.MONGODB_CONFIG
        )
        coll = mongodb_base.get_collection(collection="d_env_huizhou")

        print(f"准备插入的数据: {data}")
        insert_res = mongodb_insert_many_data(coll_conn=coll, data=data)

        if insert_res:
            LOGGER.info("页面持久化成功")
        else:
            LOGGER.error(f"页面持久化失败:{insert_res['info']}")
    except Exception as e:
        LOGGER.error(f"数据插入时发生错误：{e}")


def get_recent_data_from_db():
    """从数据库获取最近24小时的数据"""
    mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
        mongodb_config=Config.MONGODB_CONFIG
    )
    recent_data_res = mongodb_find_by_page(
        coll_conn=mongodb_base.get_collection(collection="d_env_huizhou"),
        filter_dict={},
        page=1,
        size=24,
        sorted_list=[("datetime", -1)],
        return_dict={"_id": 0, "datetime": 1},
    )
    return (
        recent_data_res.get("info", {}).get("rows", [])
        if recent_data_res.get("status")
        else []
    )


if __name__ == "__main__":
    new_data = fetch_and_process_data()
    recent_data = get_recent_data_from_db()

    # 数据去重
    recent_data_timestamps = {data["datetime"] for data in recent_data}
    unique_data = [
        data for data in new_data if data["datetime"] not in recent_data_timestamps
    ]
    print(f"新增数据:{unique_data}")

    env_data2mongodb(unique_data)
