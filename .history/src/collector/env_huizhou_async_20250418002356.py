"""
Created at 2024/03/17
Description: 惠州空气质量数据异步采集
"""

import asyncio

from datetime import datetime
from typing import Dict, Optional

import aiohttp

from src.config import LOGGER, Config
from src.databases import MongodbBase, MongodbManager


class AsyncHuiZhouAQISpider:
    """异步获取惠州的空气质量数据"""

    start_url = "https://air.cnemc.cn:18007/HourChangesPublish/GetCityRealTimeAqiHistoryByCondition?citycode=441300"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.mongodb: Optional[MongodbBase] = None

    async def init_session(self):
        """初始化HTTP会话"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    def init_mongo(self):
        """初始化MongoDB连接"""
        if not self.mongodb:
            self.mongodb = MongodbManager.get_mongodb_base(
                mongodb_config=Config.MONGODB_CONFIG
            )

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()

    async def fetch_data(self) -> Optional[Dict]:
        """异步获取城市空气质量数据"""
        try:
            await self.init_session()
            async with self.session.post(self.start_url) as response:
                if response.status == 200:
                    city_data = await response.json()
                    if city_data is None:
                        LOGGER.info("城市没有数据")
                        return None

                    return self.process_data(city_data)
                else:
                    LOGGER.error(f"获取城市数据失败，状态码：{response.status}")
                    return None
        except Exception as e:
            LOGGER.error(f"获取城市数据时发生错误：{e}")
            return None

    def process_data(self, city_data: Dict) -> Dict:
        """处理数据"""
        return {
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

    def save_to_mongodb(self, data: Dict) -> bool:
        """保存数据到MongoDB（同步操作）"""
        try:
            self.init_mongo()
            collection = self.mongodb.get_collection(collection="d_env_city")

            result = collection.insert_one(data)
            LOGGER.info(f"数据保存成功，ID: {result.inserted_id}")
            return True
        except Exception as e:
            LOGGER.error(f"保存数据时发生错误：{e}")
            return False
