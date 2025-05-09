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

        print(prediction_results)
        prediction_results = prediction_results.iloc[1:].reset_index(drop=True)

        # 格式化预测结果为字典列表
        formatted_results = prediction_results.to_dict(orient="records")

        result = {
            ResponseField.DATA: formatted_results,
            ResponseField.INFO: ResponseReply.SUCCESS,
            ResponseField.STATUS: ResponseCode.SUCCESS,
        }

        print(formatted_results)
        # 返回格式化后的结果
        return response_handle(request=current_app, dict_value=result)

    except Exception as e:
        result = {
            ResponseField.DATA: {},
            ResponseField.INFO: str(e),
            ResponseField.STATUS: ResponseCode.UNKNOWN_ERR,
        }
        return response_handle(request=current_app, dict_value=result)
