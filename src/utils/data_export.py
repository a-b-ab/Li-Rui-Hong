"""
Created by a-b-ab at 2025-05-14.
Description: 数据导出脚本
Changelog: all notable changes to this file will be documented
"""

import json

from src.databases import MongodbBase, MongodbManager


def data2json():
    """将数据库所有数据导出为json"""
    from src.config import Config

    mongodb_base: MongodbBase = MongodbManager.get_mongodb_base(
        mongodb_config=Config.MONGODB_CONFIG
    )
    collection = mongodb_base.get_collection(collection="d_aqi_huizhou")

    documents = collection.find({})

    # 将文档转换为列表
    data_list = list(documents)

    # 将 ObjectId 转换为字符串（如果存在）
    for doc in data_list:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

    # 获取当前目录并设置输出文件路径
    current_dir = Config.ROOT_DIR
    output_file = f"{current_dir}/src/AQI_display/d_aqi_huizhou.json"

    # 将数据写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"数据已导出到 {output_file}")


if __name__ == "__main__":
    data2json()
