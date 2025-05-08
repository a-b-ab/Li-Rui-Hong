"""
Created at crow 2024/04/22
Description: 惠州空气质量数据异步采集定时任务
"""

import asyncio
import json
import re

from datetime import datetime

from aiohttp import ClientSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from lxml import html

from src.config import LOGGER, Config
from src.databases import MongodbManager, mongodb_find_by_page, mongodb_insert_many_data


class AsyncGetHuiZhouAQISpider:
    """异步获取惠州的空气质量数据"""

    def __init__(self):
        self.session = ClientSession()

    async def fetch_data(self):
        """异步获取数据"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
        }

        url = "https://air.cnemc.cn:18007/HourChangesPublish/GetCityRealTimeAqiHistoryByCondition?citycode=441300"

        try:
            async with self.session.post(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data_list = await resp.json()
                    return data_list
                else:
                    LOGGER.error(f"请求失败，状态码: {resp.status}")
                    return []
        except Exception as e:
            LOGGER.error(f"请求出错: {e}")
            return []

    async def process_air_quality_data(self, data_list):
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
                LOGGER.error(f"处理数据时发生错误: {e}")
                continue

        return processed_data

    async def close(self):
        """关闭会话"""
        await self.session.close()


class AsyncGetHuiZhouHAPSpider:
    """异步获取惠州23小时大气压数据"""

    def __init__(self):
        self.splash_url = "http://0.0.0.0:8050/render.html"
        self.session = ClientSession()

    async def fetch_and_process_data(self):
        """异步获取并处理数据"""
        params = {
            "url": "https://datashareclub.com/weather/%E5%B9%BF%E4%B8%9C/%E6%83%A0%E5%B7%9E/101280301.html",  # 目标网页
            "wait": 3,  # 等待页面加载的时间（秒）
        }

        try:
            async with self.session.get(
                self.splash_url, params=params, timeout=20
            ) as response:
                if response.status == 200:
                    page_content = await response.text()
                    tree = html.fromstring(page_content)
                    data_content = tree.xpath(
                        '//script[contains(text(),"var aqi_data")]/text()'
                    )
                    aqi_data = (
                        re.findall(
                            r"var aqi_data = (\[.*?\]);", data_content[0], re.DOTALL
                        )
                        if data_content
                        else []
                    )
                    qy_data = (
                        re.findall(
                            r"var qy_data = (\[.*?\]);", data_content[0], re.DOTALL
                        )
                        if data_content
                        else []
                    )

                    aqi_data = json.loads(aqi_data[0].replace(",]", "]"))
                    qy_data = json.loads(qy_data[0].replace(",]", "]"))

                    data_list = [
                        {
                            "time_point": int(
                                datetime.strptime(aqi[0], "%Y-%m-%d %H:%M").timestamp()
                            ),
                            "hap": qy[1],
                        }
                        for aqi, qy in zip(aqi_data, qy_data)
                    ]
                    return data_list
                else:
                    LOGGER.error(f"请求失败，状态码: {response.status}")
                    return []
        except Exception as e:
            LOGGER.error(f"请求出错: {e}")
            return []

    async def close(self):
        """关闭会话"""
        await self.session.close()


async def merge_data(data_source_1, data_source_2):
    """异步合并两个数据源的数据，将 data_source_2 的 hap 字段补充到 data_source_1 中。"""
    data_source_2_dict = {item["time_point"]: item["hap"] for item in data_source_2}
    for item in data_source_1:
        time_point = item["time_point"]
        if time_point in data_source_2_dict:
            item["hap"] = data_source_2_dict[time_point]
        else:
            item["hap"] = None
    return data_source_1


async def env_data2mongodb(data):
    """异步将城市空气质量数据存入mongodb"""
    try:
        mongodb_base = MongodbManager.get_mongodb_base(
            mongodb_config=Config.MONGODB_CONFIG
        )
        coll = mongodb_base.get_collection(collection="d_env_huizhou")
        insert_res = mongodb_insert_many_data(coll_conn=coll, data=data)

        if insert_res:
            LOGGER.info("数据持久化成功")
        else:
            LOGGER.error(f"数据持久化失败:{insert_res['info']}")
    except Exception as e:
        LOGGER.error(f"数据插入时发生错误：{e}")


async def get_recent_data_from_db():
    """从数据库获取最近23小时的数据"""
    mongodb_base = MongodbManager.get_mongodb_base(mongodb_config=Config.MONGODB_CONFIG)
    recent_data_res = mongodb_find_by_page(
        coll_conn=mongodb_base.get_collection(collection="d_env_huizhou"),
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


async def collect_air_quality_data():
    """异步采集空气质量数据"""
    try:
        LOGGER.info(f"开始执行空气质量数据采集任务 - {datetime.now()}")

        aqi_spider = AsyncGetHuiZhouAQISpider()
        hap_spider = AsyncGetHuiZhouHAPSpider()

        # 异步获取数据
        city_data = await aqi_spider.fetch_data()
        aqi_data = await aqi_spider.process_air_quality_data(city_data)
        hap_data = await hap_spider.fetch_and_process_data()

        # 合并数据
        merged_data = await merge_data(aqi_data, hap_data)

        recent_data = await get_recent_data_from_db()

        # 提取 recent_data 中的 time_point 列表
        recent_time_points = [item["time_point"] for item in recent_data]

        # 数据去重
        unique_data = [
            data for data in merged_data if data["time_point"] not in recent_time_points
        ]

        # 存储到 MongoDB
        await env_data2mongodb(unique_data)

        LOGGER.info("数据采集和存储成功完成")

        # 关闭会话
        await aqi_spider.close()
        await hap_spider.close()

    except Exception as e:
        LOGGER.error(f"任务执行出错: {e}")


async def run_scheduler():
    """运行异步定时任务调度器"""
    try:
        scheduler = AsyncIOScheduler()

        # 添加定时任务 - 每天 0:00 和 12:00 执行
        scheduler.add_job(
            collect_air_quality_data,
            trigger=CronTrigger(hour="0,12", minute=0),  # 设置为每天 0:00 和 12:00
            id="air_quality_task",
            name="惠州空气质量数据采集",
            replace_existing=True,
        )

        LOGGER.info("定时任务已启动，将在每天 0:00 和 12:00 执行")
        scheduler.start()

        # 保持程序运行
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("正在关闭定时任务...")
            scheduler.shutdown()

    except Exception as e:
        LOGGER.error(f"启动定时任务失败: {e}")


if __name__ == "__main__":
    asyncio.run(run_scheduler())
