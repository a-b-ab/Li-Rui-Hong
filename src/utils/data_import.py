"""
Created by a-b-ab at 2025-05-14.
Description: 数据导入脚本
Changelog: all notable changes to this file will be documented
"""

import json
import os

from src.config import Config
from src.databases import MongodbBase, MongodbManager


def json2data():
    """将 JSON 文件中的数据导入到数据库"""

    # 获取当前目录并设置输入文件路径
    current_dir = Config.ROOT_DIR
    input_file = os.path.join(current_dir, "src\AQI_display\d_aqi_huizhou.json")

    # 读取 JSON 文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"未找到文件: {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON 文件解析错误: {e}")
        return

    # 获取 MongoDB 集合
    mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
        mongodb_config=Config.MONGODB_CONFIG
    )
    collection = mongodb_base.get_collection(collection="d_aqi_huizhou")

    # 插入数据到集合
    try:
        if data_list:
            collection.insert_many(data_list)
            print(f"成功导入 {len(data_list)} 条数据到集合 'd_aqi_huizhou'")
        else:
            print("JSON 文件中没有数据可导入")
    except Exception as e:
        print(f"数据导入失败: {e}")


if __name__ == "__main__":
    json2data()
