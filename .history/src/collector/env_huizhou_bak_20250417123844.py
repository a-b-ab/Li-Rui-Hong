"""
Created by crow at 2024-12-09.
Description: 获取所有城市的空气质量数据
Changelog: all notable changes to this file will be documented
"""

from datetime import datetime

import requests

from src.config import LOGGER, Config
from src.databases import MongodbBase, MongodbManager, mongodb_insert_many_data


class GetHuiZhouAQISpider:
    """获取惠州的空气质量数据"""

    start_url = "https://air.cnemc.cn:18007/HourChangesPublish/GetCityRealTimeAqiHistoryByCondition?citycode=441300"

    def __init__(self):
        self.city_info = {}
        self.session = requests.Session()

    def fetch_data(self):
        """获取城市空气质量数据"""
        try:
            response = self.session.post(self.start_url, timeout=10)
            if response.status_code == 200:
                city_data = response.json()
                if city_data is None:
                    LOGGER.info("城市没有数据")
                    return {}
                else:
                    city_info = {
                        "id": city_data["Id"],
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
                        "Quality": city_data["Quality"],
                        "primarypollutant": city_data["PrimaryPollutant"],
                        "measure": city_data["Measure"],
                        "unheathful": city_data["Unheathful"],
                    }
                    return city_info
            else:
                LOGGER.info(f"获取城市数据失败，状态码：{response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"获取城市数据时发生错误：{e}")
            return {}


# def env_data2mongodb(data: dict):
#     """将城市空气质量数据存入mongodb"""
#     try:
#         mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
#             mongodb_config=Config.MONGODB_CONFIG
#         )
#         coll = mongodb_base.get_collection(collection="d_env_city")

#         insert_res = mongodb_insert_many_data(coll_conn=coll, data=[data])

#         if insert_res:
#             LOGGER.info("页面持久化成功")
#         else:
#             LOGGER.error(f"页面持久化失败:{insert_res['info']}")
#     except Exception as e:
#         LOGGER.error(f"数据插入时发生错误：{e}")


def run_spider():
    """运行爬虫"""
    spider = GetHuiZhouAQISpider()
    city_data = spider.fetch_data()
    if city_data:
        print(city_data)


if __name__ == "__main__":
    run_spider()
