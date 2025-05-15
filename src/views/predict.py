"""
Created by lrh at 2025-05-09.
Description: 预测接口
Changelog: all notable changes to this file will be documented
"""

from flask import current_app
from flask_cors import cross_origin

from src.AQI_display.predict import main as aqi_predict  # 导入AQI_display的predict函数
from src.common import ResponseCode, ResponseField, ResponseReply, response_handle


@cross_origin()
def predict():
    """预测接口"""
    try:
        prediction_results = aqi_predict()

        predictions_list = prediction_results.reset_index().to_dict(orient="records")

        for record in predictions_list:
            record["time"] = record["index"].strftime("%Y-%m-%d %H:%M:%S")
            del record["index"]

        result = {
            ResponseField.DATA: predictions_list,
            ResponseField.INFO: ResponseReply.SUCCESS,
            ResponseField.STATUS: ResponseCode.SUCCESS,
        }

        # 返回格式化后的结果
        return response_handle(request=current_app, dict_value=result)

    except Exception as e:
        result = {
            ResponseField.DATA: {},
            ResponseField.INFO: str(e),
            ResponseField.STATUS: ResponseCode.UNKNOWN_ERR,
        }
        return response_handle(request=current_app, dict_value=result)
