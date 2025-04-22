"""
Created by a-b-ab at 2024-12-11.
Description: 数据修复脚本
Changelog: all notable changes to this file will be documented
"""

from datetime import datetime

from src.config import Config
from src.databases import MongodbBase, MongodbManager, mongodb_update_data


def convert_time_point_to_timestamp():
    """将数据库的timepoint字段改为时间戳"""

    mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
        mongodb_config=Config.MONGODB_CONFIG
    )
    collection = mongodb_base.get_collection(collection="d_aqi_huizhou")

    documents = collection.find({})

    for doc in documents:
        time_point = doc.get("time_point")
        if isinstance(time_point, str):
            try:
                # 将 timepoint 字段转换为时间戳，只保留到小时
                time_point_dt = datetime.strptime(time_point, "%Y-%m-%d %H")
            except ValueError:
                continue
        elif isinstance(time_point, datetime):

            time_point_dt = time_point.replace(minute=0, second=0, microsecond=0)
        else:
            continue

        timestamp = int(time_point_dt.timestamp())

        # 更新文档中的 timepoint 字段
        filter_dict = {"_id": doc["_id"]}
        update_data = {"$set": {"time_point": timestamp}}
        mongodb_update_data(collection, filter_dict, update_data)


def delete_city_code():
    """将数据库中city_code字段去掉"""
    mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
        mongodb_config=Config.MONGODB_CONFIG
    )
    collection = mongodb_base.get_collection(collection="d_aqi_huizhou")
    documents = collection.find({})
    for doc in documents:
        if "city_code" in doc:  # 检查是否存在 city_code 字段
            filter_dict = {"_id": doc["_id"]}
            update_data = {"$unset": {"city_code": ""}}  # 使用 $unset 删除字段
            mongodb_update_data(collection, filter_dict, update_data)
    print("已删除文档中的 city_code 字段")


if __name__ == "__main__":
    # convert_time_point_to_timestamp()
    delete_city_code()
