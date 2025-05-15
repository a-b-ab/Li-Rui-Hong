"""
Created by lrh at 2025-04-20.
Description: 爬虫接口
Changelog: all notable changes to this file will be documented
"""

from flask import current_app
from flask_cors import cross_origin

from src.collector.env_huizhou_bak import (
    GetHuiZhouAQISpider,
    GetHuiZhouHAPSpider,
    env_data2mongodb,
    get_recent_data_from_db,
    merge_data,
)
from src.common import ResponseCode, ResponseField, ResponseReply, response_handle
from src.utils.data_export import data2json


@cross_origin()
def spider():
    """爬虫执行接口
    {}
    """

    try:
        # 获取基本配置
        app_logger = current_app.config["app_logger"]

        # 实例化两个爬虫
        aqi_spider = GetHuiZhouAQISpider()
        hap_spider = GetHuiZhouHAPSpider()

        # 获取两个数据源的数据
        app_logger.info("开始获取 AQI 数据...")
        city_data = aqi_spider.fetch_data()
        aqi_data = aqi_spider.process_air_quality_data(city_data)

        app_logger.info("开始获取 HAP 数据...")
        hap_data = hap_spider.fetch_and_process_data()

        # 合并数据
        app_logger.info("开始合并数据...")
        merged_data = merge_data(aqi_data, hap_data)

        # 从数据库获取最近数据
        app_logger.info("从数据库获取最近数据...")
        recent_data = get_recent_data_from_db()

        # 提取 recent_data 中的 time_point 列表
        recent_time_points = [item["time_point"] for item in recent_data]

        # 数据去重
        app_logger.info("开始去重...")
        unique_data = [
            data for data in merged_data if data["time_point"] not in recent_time_points
        ]

        # 存储到 MongoDB
        app_logger.info("开始存储数据到 MongoDB...")
        env_data2mongodb(unique_data)

        # 获取数据后同步到预测模型
        data2json()
        app_logger.info("数据已导出到 JSON 文件")

        # 返回成功结果
        app_logger.info("爬虫执行成功！")
        result = {
            ResponseField.DATA: {},
            ResponseField.INFO: ResponseReply.SUCCESS,
            ResponseField.STATUS: ResponseCode.SUCCESS,
        }
        return response_handle(request=current_app, dict_value=result)

    except Exception as e:
        app_logger.error(f"爬虫执行失败：{e}")
        result = {
            ResponseField.DATA: {},
            ResponseField.INFO: str(e),
            ResponseField.STATUS: ResponseCode.UNKNOWN_ERR,
        }
        return response_handle(request=current_app, dict_value=result)
