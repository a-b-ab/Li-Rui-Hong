"""
    Created by crow at 2024-12-09.
    Description: 获取所有城市的空气质量数据
    Changelog: all notable changes to this file will be documented
"""

import json
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import schedule

from src.config import LOGGER, Config
from src.databases import MongodbBase, MongodbManager, mongodb_insert_many_data


class GetCityInfoSpider:
    """获取所有城市的空气质量数据"""

    start_url = (
        "https://air.cnemc.cn:18007/CityData/GetAQIDataPublishLiveInfo?cityCode={}"
    )

    def __init__(self):
        self.city_info = {}
        self.session = requests.Session()

    def fetch_data(self, citys_code):
        """获取城市空气质量数据"""
        try:
            response = self.session.get(self.start_url.format(citys_code), timeout=10)
            if response.status_code == 200:
                city_data = response.json()
                if city_data is None:
                    LOGGER.info(f"城市{citys_code}没有数据")
                    return {}
                else:
                    city_info = {
                        "id": city_data["Id"],
                        "area": city_data["Area"],
                        "time_point": int(
                            datetime.strptime(
                                city_data["TimePoint"], "%Y-%m-%dT%H:%M:%S"
                            ).timestamp()
                        ),
                        "AQI": city_data["AQI"],
                        "city_code": city_data["CityCode"],
                        "CO": city_data["CO"],
                        "NO2": city_data["NO2"],
                        "O3": city_data["O3"],
                        "PM10": city_data["PM10"],
                        "PM2_5": city_data["PM2_5"],
                        "SO2": city_data["SO2"],
                        "CO_level": city_data["COLevel"],
                        "NO2_level": city_data["NO2Level"],
                        "O3_level": city_data["O3Level"],
                        "PM10_level": city_data["PM10Level"],
                        "PM2_5_level": city_data["PM2_5Level"],
                        "SO2_level": city_data["SO2Level"],
                        "primarypollutant": city_data["PrimaryPollutant"],
                        "measure": city_data["Measure"],
                        "unheathful": city_data["Unheathful"],
                    }
                    return city_info
            else:
                LOGGER.info(
                    f"获取城市{citys_code}数据失败，状态码：{response.status_code}"
                )
                return {}
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"获取城市{citys_code}数据时发生错误：{e}")
            return {}


def read_city_codes(directory):
    """读取城市代码"""
    citys_codes = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            citys_codes.extend(data.values())
    return citys_codes


def env_data2mongodb(data: dict):
    """将城市空气质量数据存入mongodb"""
    try:
        mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
            mongodb_config=Config.MONGODB_CONFIG
        )
        coll = mongodb_base.get_collection(collection="d_env_city")

        insert_res = mongodb_insert_many_data(coll_conn=coll, data=[data])

        if insert_res:
            LOGGER.info("页面持久化成功")
        else:
            LOGGER.error(f"页面持久化失败:{insert_res['info']}")
    except Exception as e:
        LOGGER.error(f"数据插入时发生错误：{e}")


def process_city_code(city_code):
    """处理单个城市代码"""
    spider = GetCityInfoSpider()  # 每个线程创建一个新的实例
    city_info = spider.fetch_data(city_code)
    if city_info and "id" in city_info:
        env_data2mongodb(city_info)
        LOGGER.info(f"{city_info['area']}数据持久化成功")
        return True, city_code
    else:
        LOGGER.error(f"{city_code}数据持久化失败")
        return False, city_code


def run_spider():
    """运行爬虫"""
    city_codes = read_city_codes("data/city_code")
    success_count = 0
    failed_count = 0
    failed_city_codes = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_city_code, city_code) for city_code in city_codes
        ]
        for future in as_completed(futures):
            success, city_code = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_city_codes.append(city_code)

    LOGGER.info(f"成功持久化{success_count}条数据")
    LOGGER.info(f"失败持久化{failed_count}条数据")
    LOGGER.info(f"失败城市代码:{failed_city_codes}")


if __name__ == "__main__":
    # 使用schedule库定时执行任务(每小时)
    # schedule.every().hour.do(run_spider)

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

    # 直接执行任务
    run_spider()
