import json
import re

import requests

from lxml import html

# Splash 服务的 URL
splash_url = "http://0.0.0.0:8050/render.html"

# 请求参数
params = {
    "url": "https://datashareclub.com/weather/%E5%B9%BF%E4%B8%9C/%E6%83%A0%E5%B7%9E/101280301.html",  # 目标网页
    "wait": 2,  # 等待页面加载的时间（秒）
}

# 发送请求给 Splash
response = requests.get(splash_url, params=params, timeout=20)

# 获取页面内容
if response.status_code == 200:
    page_content = response.text  # 获取 HTML 内容
    if page_content:
        tree = html.fromstring(page_content)
        # 使用 XPath 提取包含js变量的script标签
        data_content = tree.xpath('//script[contains(text(),"var aqi_data")]/text()')
        aqi_data = (
            re.findall(r"var aqi_data = (\[.*?\]);", data_content[0], re.DOTALL)
            if data_content
            else []
        )
        qy_data = (
            re.findall(r"var qy_data = (\[.*?\]);", data_content[0], re.DOTALL)
            if data_content
            else []
        )

        # 数据清洗
        aqi_data = json.loads(aqi_data[0].replace(",]", "]"))
        qy_data = json.loads(qy_data[0].replace(",]", "]"))

        # 数据提取
        data_list = [
            {"datetime": aqi[0], "aqi": aqi[1], "hap": qy[1]}
            for aqi, qy in zip(aqi_data, qy_data)
        ]
        print(data_list)
    else:
        print("页面内容为空")

else:
    print("请求失败:", response.status_code)
