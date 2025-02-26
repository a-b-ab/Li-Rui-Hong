"""
    Created by crow at 2024-12-05.
    Description: 实现单例模式
    Changelog: all notable changes to this file will be documented
"""

from functools import wraps

from src.utils.tools import md5_encryption


def singleton(cls):
    """
    单例模式
    :param cls:cls
    :return:instance
    """
    _instance = {}

    @wraps(cls)
    def instance(*args, **kw):
        key = md5_encryption(f"{str(cls)}_{str(args)}_{str(kw)}")
        if key not in _instance:
            _instance[key] = cls(*args, **kw)
            return _instance[key]

    return instance
