"""
Created by lrh at 2024-12-09.
Description: Flask 蓝图
Changelog: all notable changes to this file will be documented
"""

from flask import Blueprint

from src.views.aqi_trend import aqi_trend
from src.views.get import get
from views.ping import ping

bp_api = Blueprint("bp_api", __name__)

bp_api.add_url_rule("/ping", view_func=ping, methods=["GET"])
bp_api.add_url_rule("/get", view_func=get, methods=["POST"])
bp_api.add_url_rule("/aqi_trend", view_func=aqi_trend, methods=["POST"])
