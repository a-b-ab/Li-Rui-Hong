"""
Created at 2024/03/17
Description: 惠州空气质量数据异步采集定时任务
"""

import asyncio

from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.collector.env_huizhou_async import AsyncHuiZhouAQISpider
from src.config import LOGGER


async def collect_air_quality_data():
    """异步采集空气质量数据"""
    try:
        LOGGER.info(f"开始执行空气质量数据采集任务 - {datetime.now()}")

        spider = AsyncHuiZhouAQISpider()

        # 异步获取数据
        data = await spider.fetch_data()

        if data:
            # 同步存储数据
            spider.save_to_mongodb(data)
            LOGGER.info("数据采集和存储成功完成")
        else:
            LOGGER.error("未获取到数据")

        # 关闭异步连接
        await spider.close()

    except Exception as e:
        LOGGER.error(f"任务执行出错: {str(e)}")


async def run_scheduler():
    """运行异步定时任务调度器"""
    try:
        scheduler = AsyncIOScheduler()

        # 添加定时任务 - 每天04:30执行
        scheduler.add_job(
            collect_air_quality_data,
            trigger=CronTrigger(hour=4, minute=30),
            id="air_quality_task",
            name="惠州空气质量数据采集",
            replace_existing=True,
        )

        LOGGER.info("定时任务已启动，将在每天04:30执行")
        scheduler.start()

        # 保持程序运行
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("正在关闭定时任务...")
            scheduler.shutdown()

    except Exception as e:
        LOGGER.error(f"启动定时任务失败: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_scheduler())
