"""
    Created by crow at 2024-12-14.
    Description: aqi趋势查询接口
    Changelog: all notable changes to this file will be documented
"""

from flask import current_app, request
from flask_cors import cross_origin

from src.common import (
    ResponseCode,
    ResponseField,
    ResponseReply,
    UniResponse,
    response_handle,
)
from src.config import Config
from src.databases import MongodbBase, mongodb_find_by_page


@cross_origin()
def aqi_trend():
    """查询接口
    {
        "area":"惠州市",
    }

    """

    # 获取基本配置
    app_logger = current_app.config["app_logger"]
    mongodb_base: MongodbBase = current_app.config["mongodb_base"]
    _: Config = current_app.config["app_config"]

    # 获取基础数据
    post_data = request.json
    area = post_data.get("area")
    time_list = post_data.get("time_list")
    size: int = post_data.get("size", 24)

    try:
        page: int = int(post_data.get("page", 1))
    except Exception:
        page = 1

    if page < 1 or page > 1000:
        # 超出限制
        result = {
            ResponseField.DATA: {},
            ResponseField.INFO: "分页参数超出限制",
            ResponseField.STATUS: 1010,
        }
        app_logger.error(f"API {request.path} 分页参数超出限制")

    filter_dict = {}

    if area:
        filter_dict["area"] = {"$regex": area}
    if time_list and len(time_list) == 2:
        start_time, end_time = time_list
        filter_dict["time_point"] = {"$gte": start_time, "lte": end_time}

    # 获取配置信息
    find_db_res = mongodb_find_by_page(
        coll_conn=mongodb_base.get_collection(collection="d_env_city"),
        filter_dict=filter_dict,
        size=size,
        page=page,
        sorted_list=[("time_point", -1)],
        return_dict={"_id": 0, "AQI": 1, "time_point": 1},
    )

    find_db_data = find_db_res["info"]
    if find_db_res["status"]:
        result = {
            ResponseField.DATA: find_db_data,
            ResponseField.INFO: ResponseReply.SUCCESS,
            ResponseField.STATUS: ResponseCode.SUCCESS,
        }
        app_logger.info(f"API {request.path} 查询成功")
    else:
        result = {
            **UniResponse.DB_ERR,
            **{ResponseField.DATA: {"err_msg": find_db_data}},
        }
        app_logger.error(f"API {request.path} 查询失败")

    return response_handle(request=request, dict_value=result)
