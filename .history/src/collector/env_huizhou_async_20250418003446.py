"""
Created at 2024/03/17
Description: 惠州空气质量数据异步采集
"""

import asyncio
import json
import re

from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

from lxml import html

from src.config import LOGGER, Config
from src.databases import (
    MongodbBase,
    MongodbManager,
    mongodb_find_by_page,
    mongodb_insert_many_data,
)


class AsyncHuiZhouAQISpider:
    """异步获取惠州的空气质量数据"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.url = "https://air.cnemc.cn:18007/HourChangesPublish/GetCityRealTimeAqiHistoryByCondition?citycode=441300"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
        }

    async def init_session(self):
        """初始化HTTP会话"""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()

    async def fetch_data(self) -> List[Dict]:
        """异步获取数据"""
        try:
            await self.init_session()
            async with self.session.post(self.url, ssl=False) as response:
                if response.status == 200:
                    data_list = await response.json()
                    LOGGER.info("AQI数据获取成功")
                    return data_list
                else:
                    LOGGER.error(f"获取AQI数据失败，状态码：{response.status}")
                    return []
        except Exception as e:
            LOGGER.error(f"获取AQI数据时发生错误：{e}")
            return []

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
                LOGGER.error(f"处理AQI数据时发生错误：{e}")
                continue

        return processed_data


class AsyncHuiZhouHAPSpider:
    """异步获取惠州23小时大气压数据"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.splash_url = "http://0.0.0.0:8050/render.html"
        self.target_url = "https://datashareclub.com/weather/%E5%B9%BF%E4%B8%9C/%E6%83%A0%E5%B7%9E/101280301.html"

    async def init_session(self):
        """初始化HTTP会话"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()

    async def fetch_and_process_data(self) -> List[Dict]:
        """异步获取并处理数据"""
        try:
            await self.init_session()
            params = {
                "url": self.target_url,
                "wait": 3,
            }

            async with self.session.get(self.splash_url, params=params) as response:
                if response.status == 200:
                    page_content = await response.text()
                    return self._process_html_content(page_content)
                else:
                    LOGGER.error(f"获取HAP数据失败，状态码：{response.status}")
                    return []
        except Exception as e:
            LOGGER.error(f"获取HAP数据时发生错误：{e}")
            return []

    def _process_html_content(self, page_content: str) -> List[Dict]:
        """处理HTML内容"""
        try:
            if not page_content:
                return []

            tree = html.fromstring(page_content)
            data_content = tree.xpath(
                '//script[contains(text(),"var aqi_data")]/text()'
            )

            if not data_content:
                return []

            aqi_data = re.findall(
                r"var aqi_data = (\[.*?\]);", data_content[0], re.DOTALL
            )
            qy_data = re.findall(
                r"var qy_data = (\[.*?\]);", data_content[0], re.DOTALL
            )

            if not aqi_data or not qy_data:
                return []

            # 数据清洗
            aqi_data = json.loads(aqi_data[0].replace(",]", "]"))
            qy_data = json.loads(qy_data[0].replace(",]", "]"))

            # 数据提取
            return [
                {
                    "time_point": int(
                        datetime.strptime(aqi[0], "%Y-%m-%d %H:%M").timestamp()
                    ),
                    "hap": qy[1],
                }
                for aqi, qy in zip(aqi_data, qy_data)
            ]
        except Exception as e:
            LOGGER.error(f"处理HAP数据时发生错误：{e}")
            return []


class DataManager:
    """数据管理类"""

    @staticmethod
    def merge_data(data_source_1: List[Dict], data_source_2: List[Dict]) -> List[Dict]:
        """合并两个数据源"""
        data_source_2_dict = {item["time_point"]: item["hap"] for item in data_source_2}

        for item in data_source_1:
            time_point = item["time_point"]
            item["hap"] = data_source_2_dict.get(time_point)

        return data_source_1

    @staticmethod
    def get_recent_data_from_db() -> List[Dict]:
        """从数据库获取最近数据"""
        try:
            mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
                mongodb_config=Config.MONGODB_CONFIG
            )
            recent_data_res = mongodb_find_by_page(
                coll_conn=mongodb_base.get_collection(collection="d_aqi_huizhou"),
                filter_dict={},
                page=1,
                size=23,
                sorted_list=[("time_point", -1)],
                return_dict={"_id": 0, "time_point": 1},
            )
            return (
                recent_data_res.get("info", {}).get("rows", [])
                if recent_data_res.get("status")
                else []
            )
        except Exception as e:
            LOGGER.error(f"获取最近数据时发生错误：{e}")
            return []

    @staticmethod
    def save_to_mongodb(data: List[Dict]) -> bool:
        """保存数据到MongoDB"""
        try:
            if not data:
                LOGGER.info("没有新数据需要保存")
                return True

            mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
                mongodb_config=Config.MONGODB_CONFIG
            )
            coll = mongodb_base.get_collection(collection="d_aqi_huizhou")

            insert_res = mongodb_insert_many_data(coll_conn=coll, data=data)

            if insert_res:
                LOGGER.info(f"成功保存 {len(data)} 条数据")
                return True
            else:
                LOGGER.error(f"数据保存失败:{insert_res['info']}")
                return False
        except Exception as e:
            LOGGER.error(f"保存数据时发生错误：{e}")
            return False


async def run_spider():
    """运行爬虫主函数"""
    try:
        # 创建爬虫实例
        aqi_spider = AsyncHuiZhouAQISpider()
        hap_spider = AsyncHuiZhouHAPSpider()

        # 并发获取两个数据源的数据
        aqi_data_raw, hap_data = await asyncio.gather(
            aqi_spider.fetch_data(), hap_spider.fetch_and_process_data()
        )

        # 关闭爬虫连接
        await asyncio.gather(aqi_spider.close(), hap_spider.close())

        # 处理AQI数据
        aqi_data = aqi_spider.process_air_quality_data(aqi_data_raw)

        # 合并数据
        merged_data = DataManager.merge_data(aqi_data, hap_data)

        # 获取最近数据
        recent_data = DataManager.get_recent_data_from_db()
        recent_time_points = {item["time_point"] for item in recent_data}

        # 数据去重
        unique_data = [
            data for data in merged_data if data["time_point"] not in recent_time_points
        ]

        # 保存到MongoDB
        if DataManager.save_to_mongodb(unique_data):
            LOGGER.info("数据采集任务完成")
        else:
            LOGGER.error("数据采集任务失败")

    except Exception as e:
        LOGGER.error(f"爬虫运行出错:{e}")
