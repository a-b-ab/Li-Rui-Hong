"""
    Created by crow at 2024-12-10.
    Description: 校验中间件
    Changelog: all notable changes to this file will be documented
"""

from functools import wraps

from flask import request

from src.config import LOGGER, Config

from .response_base import UniResponse, response_handle


def token_required():
    """Token 校验装饰器"""

    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            if request.method == "POST":
                app_id = request.headers.get("MS_APP_ID", "")
                app_token = request.headers.get("MS_APP_TOKEN", "")
                if app_id and app_token and app_id in Config.APP_ID_CONFIG:
                    if app_token == Config.APP_ID_CONFIG[app_id]:
                        resp = exec_decorator_fn(fn, *args, **kwargs)
                    else:
                        resp = return_401()
                else:
                    resp = return_401()
            else:
                resp = return_401()
            return resp

        return decorator

    return wrapper


def auth_post_params(key: list = None, data: list = None, allow_empty: bool = True):
    """
    :param keys: list
    :param data: list
    :param allow_empty:bool
    :return:
    """

    keys, data = key or [], data or []

    def wrapper(func):
        @wraps(func)
        def auth_action_param(*args, **kwargs):
            """
            接口验证装饰器
            """
            post_data = request.json
            request_keys = post_data.keys()
            allow_empty_valid = True
            if not allow_empty:
                # post_data 字典所有key对应value不能为空
                for key in keys:
                    if not post_data.get(key):
                        LOGGER.error(
                            f"request parameter error:{request.path} - {post_data}"
                        )
                        allow_empty_valid = False
                        break

            if set(keys).issubset(set(request_keys)) and allow_empty_valid:
                if "data" in request_keys and data:
                    data_keys = post_data["data"].keys()
                    if set(data).issubset(set(data_keys)):
                        resp = func(*args, **kwargs)
                    else:
                        LOGGER.error(
                            f"request parameter error:{request.path} - {post_data}"
                        )
                        resp = return_params_error()
                else:
                    resp = func()

            else:
                LOGGER.error(f"request parameter error:{request.path} - {post_data}")
                resp = return_params_error()
            return resp

        return auth_action_param

    return wrapper


def exec_decorator_fn(decorator, *args, **kwargs):
    """
    执行装饰器函数
    :param decorator_fn: 装饰器函数
    :param args: 参数
    :param kwargs: 关键字参数
    :return: 装饰器函数执行结果
    """
    try:
        resp = decorator(*args, **kwargs)
    except Exception as e:
        resp = UniResponse.SERVER_UNKNOWN_ERR
        LOGGER.exception(f"请求{request.path}出错,{e}")
    return resp


def return_params_error():
    """
    参数错误返回
    :return: 参数错误返回
    """
    return response_handle(
        request=request,
        dict_value=UniResponse.PARAM_ERR,
    )


def return_401():
    """
    401返回
    :return: 401返回
    """
    return response_handle(
        request=request,
        dict_value=UniResponse.NOT_AUTHORIZED,
        status=401,
    )
