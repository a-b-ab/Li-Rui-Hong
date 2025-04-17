### 惠州空气质量和大气压查询接口

数据库
d_aqi_huizhou

#### 请求 URL

`localhost:8765/get`

#### 请求方法

`POST`

#### 请求头

- `Content-Type: application/json`

#### 请求参数

| 参数名称       | 类型        | 是否必填 | 默认值 | 描述                                       |
|------------       |--------      |----------    |--------    |---------------------                |
| time_list               | list     |         否       | [1111,1111]          | 基于时间段查询 |

#### 请求示例

```json
{
        "time_list":[1740470400,1740481200]
    }
        
```

    #### 返回示例
    
    
    
```json
{
  "data": {
    "total": 4,
    "rows": [
      {
        "datetime": 1740470400,
        "aqi": 52,
        "hap": 1011,
        "area": "惠州"
      },
      {
        "datetime": 1740474000,
        "aqi": 49,
        "hap": 1011,
        "area": "惠州"
      },
      {
        "datetime": 1740477600,
        "aqi": 48,
        "hap": 1011,
        "area": "惠州"
      },
      {
        "datetime": 1740481200,
        "aqi": 51,
        "hap": 1012,
        "area": "惠州"
      }
    ]
  },
  "info": "ok",
  "status": 0
}
        
```

> 参数错误

```
{
    "data": {},
    "info": "参数错误",
    "status": 100
}
```

> 数据库错误

```
{
    "data": {
    "err_msg": {}
    },
    "info": "数据库错误",
    "status": 106
}
```    
  
  