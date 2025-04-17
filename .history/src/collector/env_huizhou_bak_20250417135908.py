"""
Created by crow at 2024-12-09.
Description: 获取所有城市的空气质量数据
Changelog: all notable changes to this file will be documented
"""

from datetime import datetime
from typing import Dict, List

import requests

from src.config import LOGGER, Config
from src.databases import MongodbBase, MongodbManager, mongodb_insert_many_data


class GetHuiZhouAQISpider:
    """获取惠州的空气质量数据"""

    def __init__(self):
        self.session = requests.Session()

    def process_air_quality_data(self, data_list: List[Dict]) -> List[Dict]:
        """处理空气质量数据"""
        processed_data = []
        for item in data_list:
            try:
                # 处理时间戳 - 从/Date(1744783200000)/格式转换
                time_str = item["TimePoint"]
                timestamp = (
                    int(time_str.replace("/Date(", "").replace(")/", "")) // 1000
                )

                city_info = {
                    "id": item["Id"],
                    "time_point": timestamp,
                    "time_point_str": item["TimePointStr"],
                    "AQI": item["AQI"],
                    "city_code": item["CityCode"],
                    "CO": item["CO"],
                    "NO2": item["NO2"],
                    "O3": item["O3"],
                    "PM10": item["PM10"],
                    "PM2_5": item["PM2_5"],
                    "SO2": item["SO2"],
                    "Quality": item["Quality"],
                    "primarypollutant": item["PrimaryPollutant"],
                    "measure": item["Measure"],
                    "unheathful": item["Unheathful"],
                }
                processed_data.append(city_info)
            except Exception as e:
                LOGGER.error(f"处理数据时发生错误：{e}")
                continue

        return processed_data


def env_data2mongodb(data_list: List[Dict]):
    """将城市空气质量数据存入mongodb"""
    try:
        mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
            mongodb_config=Config.MONGODB_CONFIG
        )
        coll = mongodb_base.get_collection(collection="d_env_city")

        insert_res = mongodb_insert_many_data(coll_conn=coll, data=data_list)

        if insert_res:
            LOGGER.info(f"成功存储 {len(data_list)} 条数据")
        else:
            LOGGER.error(f"数据持久化失败:{insert_res['info']}")
    except Exception as e:
        LOGGER.error(f"数据插入时发生错误：{e}")


def run_spider(data_list: List[Dict]):
    """运行数据处理和存储"""
    spider = GetHuiZhouAQISpider()
    processed_data = spider.process_air_quality_data(data_list)
    if processed_data:
        env_data2mongodb(processed_data)
        LOGGER.info("数据处理和存储完成")
    else:
        LOGGER.error("没有可处理的数据")


if __name__ == "__main__":
    # 这里放入您的24小时数据列表
    air_quality_data = []  # 您的24小时数据
    run_spider(air_quality_data)
