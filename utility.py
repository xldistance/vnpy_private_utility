"""
常用方法组件
"""
from copy import copy
from time import sleep
import csv
import json
from re import T
import sys
from pathlib import Path
from typing import (Callable, Dict,List, Union,Any,Tuple)
import logging
from uuid import uuid4
from datetime import datetime,timedelta,time
from empyrical import (annual_volatility)
from math import floor, ceil
from pytz import timezone,common_timezones
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import pandas as pd
import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import requests
import talib
from dingtalkchatbot.chatbot import DingtalkChatbot
import psutil
import redis
import zlib
import cloudpickle
import h5py
import platform

from vnpy.trader.event import EVENT_TIMER,REDIS_CLIENT,EVENT_LOG
from vnpy.event import Event,EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.object import BarData, TickData,LogData,Status
from vnpy.trader.constant import Exchange, Interval
TZ_INFO = timezone("Asia/Shanghai")
if platform.uname().system == "Windows":
    LINK_SIGN = "\\"
elif platform.uname().system == "Linux":
    LINK_SIGN = "/"
#------------------------------------------------------------------------------------
def get_index_vt_symbol(vt_symbol) -> str:
    """
    获取指数合约vt_symbol
    """
    symbol, exchange,gateway_name = extract_vt_symbol(vt_symbol)
    symbol_mark = get_symbol_mark(vt_symbol)
    index_vt_symbol = f"{symbol_mark}99" + "_" + exchange.value + "/" + gateway_name
    return index_vt_symbol
#-------------------------------------------------------------------------------------------------
def save_csv(filepath:str,data:Any):
    """
    保存数据到csv
    """
    if not Path(filepath).exists():  # 如果文件不存在，需要写header
        with open(filepath, "w", newline="",encoding="utf_8_sig") as file:  #newline=""不自动换行
            write_file = csv.DictWriter(file, list(data.__dict__.keys()))
            write_file.writeheader()
            write_file.writerow(data.__dict__)
    else:  # 文件存在，不需要写header
        with open(filepath, "a", newline="",encoding="utf_8_sig") as file:  #a追加形式写入
            write_file = csv.DictWriter(file, list(data.__dict__.keys()))
            write_file.writerow(data.__dict__)    
#------------------------------------------------------------------------------------
def save_h5(filename:str,data:Union[list,tuple,set,dict],overwrite:bool=False):
    """
    * 保存hdf5数据
    * filename文件名，data要保存的数据
    * overwrite为True覆盖源文件,为False增量更新文件
    """
    contract_file_path = get_folder_path(filename)
    filepath =f"{contract_file_path}{LINK_SIGN}{filename}.h5"
    if overwrite:
        raw_data = data
    else:
        #增量更新数据
        raw_data:Union[list,tuple,set,dict] = load_h5(filename)
        if isinstance(raw_data,dict) and isinstance(data,dict):
            raw_data.update(data)
        elif isinstance(raw_data,tuple) and isinstance(data,tuple):
            raw_data = (*raw_data,*data)
        elif isinstance(raw_data,set) and isinstance(data,set):
            raw_data = raw_data | data
        elif isinstance(raw_data,list) and isinstance(data,list):
            # 列表去重
            raw_data = list_de_duplication(raw_data + data)
    #循环写入h5数据直到写入成功或重试3次后退出循环
    count = 0
    while True:
        count += 1
        status = save_h5_status(filepath,raw_data)
        if status or count > 3:
            break
#------------------------------------------------------------------------------------
def save_h5_status(filepath:str,raw_data:Any):
    """
    获取H5保存数据状态
    """
    try:
        with h5py.File(filepath, "w") as file:
            data = zlib.compress(cloudpickle.dumps(raw_data), 5)
            file.create_dataset("data", data=np.void(data))
            #file["data"] =np.void(data)
        return True
    except Exception:
        return False
#------------------------------------------------------------------------------------
def load_h5(filename:str):
    """
    读取hdf5数据
    """
    contract_file_path = get_folder_path(filename)
    filepath =f"{contract_file_path}{LINK_SIGN}{filename}.h5"

    if not Path(filepath).exists():
        return {}
    count = 0
    while True:
        count += 1
        status,data = load_h5_status(filepath)
        if status or count > 3:
            return data
#------------------------------------------------------------------------------------        
def load_h5_status(filepath:str):
    """
    获取H5读取状态及数据
    """
    try:
        with  h5py.File(filepath,"r") as file:
            data = file["data"][()]
            data = cloudpickle.loads(zlib.decompress(data))
            return True,data
    except:
        return False,{}
#------------------------------------------------------------------------------------
def index_location(values:list):
    """
    获取列表相同值索引
    """
    index_location = {}
    for index, value in enumerate(values):
        if value not in index_location:
            index_location[value] = [index]
        else:
            index_location[value].append(index)
    return index_location
#------------------------------------------------------------------------------------
def list_de_duplication(value:List):
    """
    列表去重复且保持原顺序
    """
    return sorted(set(value),key=value.index)
#------------------------------------------------------------------------------------
def get_uuid():
    """
    获取唯一id
    """
    id_ = str(uuid4()).replace('-', '')
    return id_
#------------------------------------------------------------------------------------
def list_of_groups(init_list:list, children_list_len:int):
    """
    n等分列表
    """
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len  #取列表余数
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list
#----------------------------------------------------------------------
def dict_slice(origin_dict:dict, start:int, end:int):
    """
    1.字典切片取值
    2.origin_dict: 字典,start: 起始,end: 终点
    """
    slice_dict = {key: origin_dict[key] for key in list(origin_dict.keys())[start:end]}
    return slice_dict
#------------------------------------------------------------------------------------
def get_digits(value: float) -> int:
    """
    获取浮点数小数点后数值长度
    """
    if "e" in str(value):
        value_str = "{:.10f}".format(value)
    else:
        value_str = str(float(value))
    for index in range(10):
        value_str = delete_zero(value_str)
    _, buf = value_str.split(".")
    return len(buf)
#------------------------------------------------------------------------------------    
def delete_zero(value:str) -> str:
    """
    删除字符串数值类型末尾0
    """
    if value[-1] == "0":
        value = value[:-1]
    return value
#------------------------------------------------------------------------------------
def remain_alpha(convert_contract:str) -> str:
    """
    返回合约symbol或字符串的字母部分
    """
    if "_" in convert_contract and "/" in convert_contract:
        convert_contract = extract_vt_symbol(convert_contract)[0]
    else:
        if "_" in convert_contract:
            convert_contract = convert_contract.split("_")[0]
    symbol_mark = "".join(list(filter(str.isalpha,convert_contract)))
    return symbol_mark
#------------------------------------------------------------------------------------
def remain_digit(convert_contract:str) -> str:
    """
    返回合约symbol或字符串的数字部分
    """
    if "_" in convert_contract and "/" in convert_contract:
        convert_contract = extract_vt_symbol(convert_contract)[0]
    else:
        if "_" in convert_contract:
            convert_contract = convert_contract.split("_")[0]
    symbol_mark = "".join(list(filter(str.isdigit,convert_contract)))
    return symbol_mark
#------------------------------------------------------------------------------------
def extract_vt_symbol(vt_symbol: str) ->Tuple[str,Exchange,str]:
    """
    返回(symbol:str, exchange: Exchange,gateway_name:str)
    """
    *symbol_exchange,gateway_name = vt_symbol.split("/")
    if len(symbol_exchange) > 1:
        symbol_exchange = "/".join(symbol_exchange)
    else:
        symbol_exchange = symbol_exchange[0]
    *symbols,exchange = symbol_exchange.split("_")
    symbol = "_".join([symbols[index] for index in range(len(symbols))])
    return symbol, Exchange(exchange),gateway_name
#------------------------------------------------------------------------------------
def save_connection_status(gateway_name:str,status:bool):
    """
    保存交易接口连接状态，status为False时交易父进程会自动重启策略交易子进程
    """
    #gateway_name为空值直接返回
    if not gateway_name:
        return
    
    connection_status = load_json("connection_status.json")
    connection_status.update({gateway_name:status})
    save_json("connection_status.json",connection_status)
#------------------------------------------------------------------------------------
def save_redis_data(file_name: str,data:Any):
    """
    保存redis数据
    """
    REDIS_CLIENT.hset(file_name, file_name, zlib.compress(cloudpickle.dumps(data), 5))
    # 设置数据过期时间
    extime = datetime.now() + timedelta(days=1)
    REDIS_CLIENT.expireat(file_name, extime)
#------------------------------------------------------------------------------------
def load_redis_data(file_name: str):
    """
    读取redis数据
    """
    data = REDIS_CLIENT.hget(file_name, file_name)  #读取redis data
    if data:
        data = cloudpickle.loads(zlib.decompress(data))          # 解压还原原始数据
    return data
#------------------------------------------------------------------------------------
def generate_vt_symbol(symbol: str, exchange: Exchange,gateway_name: str):
    """
    生成vt_symbol
    """
    return f"{symbol}_{exchange.value}/{gateway_name}"
#------------------------------------------------------------------------------------
def _get_trader_dir(temp_name: str):
    """
    获取.vntrader工作路径
    """
    cwd = Path.cwd()
    temp_path = cwd.joinpath(temp_name)

    # .vntrader已存在返回当前工作路径
    if temp_path.exists():
        return cwd, temp_path

    # 否则使用系统的主路径
    home_path = Path.home()
    temp_path = home_path.joinpath(temp_name)

    # 不存在系统路径则创建.vntrader文件夹
    if not temp_path.exists():
        temp_path.mkdir()

    return home_path, temp_path

TRADER_DIR, TEMP_DIR = _get_trader_dir(".vntrader")
sys.path.append(str(TRADER_DIR))
#------------------------------------------------------------------------------------
def get_file_path(filename: str):
    """
    返回文件路径
    """
    return TEMP_DIR.joinpath(filename)
#------------------------------------------------------------------------------------
def get_folder_path(folder_name: str):
    """
    返回文件夹路径
    """
    folder_path = TEMP_DIR.joinpath(folder_name)
    if not folder_path.exists():
        folder_path.mkdir()
    return folder_path
#------------------------------------------------------------------------------------
def get_icon_path(filepath: str, ico_name: str):
    """
    返回图标路径
    """
    ui_path = Path(filepath).parent
    icon_path = ui_path.joinpath("ico", ico_name)
    return str(icon_path)
#------------------------------------------------------------------------------------
def save_json(filename: str, data: Union[List,Dict]):
    """
    保存数据到json文件
    """
    filepath = get_file_path(filename)
    try:
        with open(filepath, mode="w+", encoding="UTF-8") as file:
            json.dump(data, file, sort_keys=True, indent=4, ensure_ascii=False)
    except Exception as err:
        print(f"文件：{filename}保存数据出错，错误信息：{err}")
        return
#------------------------------------------------------------------------------------
def load_json(filename: str) ->Dict:
    """
    读取json文件
    """
    filepath = get_file_path(filename)
    if filepath.exists():
        with open(filepath, mode="r", encoding="UTF-8") as file:
            try:
                data = json.load(file)
            except:
                data = {}
        return data
    else:
        save_json(filename, {})
        return {}
#------------------------------------------------------------------------------------
#启动程序时缓存dr_data
DR_DATA = load_json("data_recorder_setting.json")
#------------------------------------------------------------------------------------
def round_to(value:Union[str, float], target: float) -> float:
    """
    取整value到最小变动
    """
    target = float(target)
    value = np.nan_to_num(value)
    try:
        rounded = int(round(value / target)) * target
    except Exception as err:
        rounded = 0
    #浮点数再次取整,防止返回数值精度不对
    if isinstance(rounded, float):
        rounded = round(rounded,get_digits(target))
    return rounded
#------------------------------------------------------------------------------------
def floor_to(value:Union[str, float], target: float) -> float:
    """
    value向下取整到最小变动
    """
    tmp = float(target)
    rounded = int(floor(np.nan_to_num(float(value)) / tmp)) * tmp    
    return rounded
#------------------------------------------------------------------------------------
def ceil_to(value:Union[str, float], target: float) -> float:
    """
    value向上取整到最小变动
    """
    tmp = float(target)
    rounded = int(ceil(np.nan_to_num(float(value)) / tmp)) * tmp    
    return rounded   
#------------------------------------------------------------------------------------
def get_local_datetime(timestamp:Union[str, float,int],hours:int = 8):
    """
    1.把timestamp或者str的datetime转换时区
    2.hours = 8:转化datetime 0时区到东八区，hours = 0:东八区datetime转化
    """
    if isinstance(timestamp,str):
        # 去除时区字符串
        if "+" in timestamp:
            timestamp = timestamp.split("+")[0]
        if "Z" in timestamp:
            if "." in timestamp:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%dT%H:%M:%SZ")
        elif "T" in timestamp:
            if "." in timestamp:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%dT%H:%M:%S.%f")
            else:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%dT%H:%M:%S")
        else:
            if "." in timestamp:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%d %H:%M:%S.%f")
            else:
                local_time = pd.to_datetime(timestamp,format="%Y-%m-%d %H:%M:%S")
    elif isinstance(timestamp,float):
        local_time = pd.to_datetime(timestamp,unit = "s")
    elif isinstance(timestamp,int):
        #秒时间戳
        if len(str(timestamp)) == 10:
            local_time = pd.to_datetime(timestamp,unit = "s")
        #毫秒时间戳
        elif len(str(timestamp)) == 13:
            local_time = pd.to_datetime(timestamp,unit = "ms")
        #微妙时间戳
        elif len(str(timestamp)) == 16:
            local_time = pd.to_datetime(timestamp,unit = "us")
        #纳秒时间戳
        elif len(str(timestamp)) == 19:
            local_time = pd.to_datetime(timestamp,unit = "ns")
    return (local_time+timedelta(hours=hours)).tz_localize(TZ_INFO)  #pandas timestamp添加时区
#------------------------------------------------------------------------------------
def add_timezone(dt):
    """
    datetime添加时区,用isinstance判定实例会堵塞,只能用try except
    """
    try:    #datetime timestamp
        dt = dt.astimezone(TZ_INFO)
    except:     #pandas timestamp
        dt = dt.tz_localize(TZ_INFO)
    return dt
#------------------------------------------------------------------------------------
def get_symbol_mark(vt_symbol:str) -> str:
    """
    获取合约标识(CTP接口合约标识区分大小写，数字货币接口合约标识都为小写)
    * "rb2201_SHFE/CTP"合约标识为"rb"，"ZC2201_CZCE/CTP"合约标识为"ZC"，"BTCUSD_BINANCES/BINANCES"合约标识为"btc"
    """
    symbol,exchange,gateway_name = extract_vt_symbol(vt_symbol)
    if remain_alpha(gateway_name) in ["CTP"]:
        #CTP合约标识
        symbol_mark = remain_alpha(vt_symbol)
    elif remain_alpha(gateway_name) in ["OKEX"] and symbol[-1].isdigit():
        #OKEX接口有币本位交割(btcusd99)和USDT交割(btcusdt99)，合成指数合约需要区分symbol_mark
        symbol_mark = remain_alpha(vt_symbol).lower()
    else:    
        #数字货币合约标识
        if "." in symbol:
            symbol_mark =symbol.split(".")[0]
        elif "USDT" in symbol:
            symbol_mark = remain_alpha(symbol.split("USDT")[0]).lower()
        elif "USD" in symbol:
            symbol_mark = remain_alpha(symbol.split("USD")[0]).lower()
        elif "PERP" in symbol:
            symbol_mark = remain_alpha(symbol.split("PERP")[0]).lower()
        else:
            #其他交易所合约标识
            symbol_mark = remain_alpha(vt_symbol).lower()
            
    return symbol_mark
#-----------------------------------------------------------------------
def quarter_date_count(now_datetime:datetime):
    """
    1.季度合约日期计算
    2.返回季度合约月份日期和第二周周五所在日
    """
    current_month = now_datetime.month
    if current_month <=3:
        target_month = 3
        target_date = datetime.strptime(f"{now_datetime.year}-04-01","%Y-%m-%d")
    elif 3 < current_month <=6:
        target_month = 6
        target_date = datetime.strptime(f"{now_datetime.year}-07-01","%Y-%m-%d")
    elif 6 < current_month <=9:
        target_month = 9
        target_date = datetime.strptime(f"{now_datetime.year}-10-01","%Y-%m-%d")
    elif 9 < current_month <=12:
        target_month = 12
        target_date = datetime.strptime(f"{now_datetime.year+1}-01-01","%Y-%m-%d")
    days_ago = (7 + target_date.weekday() - 4) % 7          #季度合约周五结算,weekday:4
    if days_ago == 0:
        days_ago = 7
    target_date -= timedelta(days_ago)
    target_year = target_date.year
    target_day = target_date.day
    return (target_year,target_month,target_day),target_day-14
#-----------------------------------------------------------------------
def get_quarter_postfix(now_datetime:datetime):
    """
    返回季度合约symbol后缀
    """
    if not now_datetime:
        now_datetime = datetime.now(TZ_INFO)
    quarter_date,second_weekday = quarter_date_count(now_datetime)
    quarter_year,quarter_month,quarter_day = quarter_date
    if quarter_year == now_datetime.year and quarter_month ==  now_datetime.month and now_datetime.day >= second_weekday and  now_datetime.hour >= 16:
        target_datetime = now_datetime+ relativedelta(months=1)
        quarter_date,second_weekday = quarter_date_count(target_datetime)
        quarter_year,quarter_month,quarter_day = quarter_date

    if quarter_month < 10:
        symbol_postfix = f"{str(quarter_year)[-2:]}0{quarter_month}{quarter_day}"
    else:
        symbol_postfix = f"{str(quarter_year)[-2:]}{quarter_month}{quarter_day}"
    return symbol_postfix
#-----------------------------------------------------------------------
def current_date_count(current_month:int):
    """
    计算月份最后一个周五所在日期
    """
    end_date = datetime.now(TZ_INFO)   
    if current_month < 12:      
        target_date = datetime.strptime(f"{end_date.year}-{current_month+1}-01","%Y-%m-%d")
    else:
        target_date = datetime.strptime(f"{end_date.year+1}-01-01","%Y-%m-%d")
    days_ago = (7 + target_date.weekday() - 4) % 7          #周五结算,weekday:4
    if days_ago == 0:
        days_ago = 7
    target_date -= timedelta(days_ago)
    target_year = target_date.year
    target_day = target_date.day
    return target_year,current_month,target_day
#-----------------------------------------------------------------------
def get_friday_postfix():
    """
    返回月份每个周五所在日期str
    """
    current_year,current_month,current_day = current_date_count(datetime.now(TZ_INFO).month)
    if datetime.now(TZ_INFO).day > current_day:
        current_year,current_month,current_day = current_date_count(datetime.now(TZ_INFO).month+1)   
    if current_month < 10:
        month_zero_mark = "0"
    else:
        month_zero_mark = ""
    postfix_1 = f"{str(current_year)[-2:]}{month_zero_mark}{current_month}0{current_day-21}"
    postfix_2 = f"{str(current_year)[-2:]}{month_zero_mark}{current_month}{current_day-14}"
    postfix_3 = f"{str(current_year)[-2:]}{month_zero_mark}{current_month}{current_day-7}"
    postfix_4 = f"{str(current_year)[-2:]}{month_zero_mark}{current_month}{current_day}"

    return postfix_1,postfix_2,postfix_3,postfix_4
#-----------------------------------------------------------------------
def get_current_next_postfix():
    """
    返回当周，次周合约后缀
    """
    postfix_1,postfix_2,postfix_3,postfix_4 = get_friday_postfix()
    if datetime.now(TZ_INFO).day < 10:
        int_now_date = int(f"{datetime.now(TZ_INFO).month}0{datetime.now(TZ_INFO).day}")
    else:
        int_now_date = int(f"{datetime.now(TZ_INFO).month}{datetime.now(TZ_INFO).day}")
    if int_now_date <= int(postfix_1[-4:]):
        current_symbol_postfix = postfix_1
        next_symbol_postfix = postfix_2
    elif int(postfix_1[-4:]) < int_now_date <=int(postfix_2[-4:]):
        current_symbol_postfix = postfix_2
        next_symbol_postfix = postfix_3
    elif int(postfix_2[-4:]) < int_now_date <=int(postfix_3[-4:]):
        current_symbol_postfix = postfix_3
        next_symbol_postfix = postfix_4   
    elif int(postfix_3[-4:]) < int_now_date <=int(postfix_4[-4:]):
        postfix_1_datetime = datetime.strptime(f"{str(datetime.now(TZ_INFO).year)[:2]}{postfix_4}","%Y%m%d") + timedelta(days = 7)
        if postfix_1_datetime.month < 10:
            zero_mark = "0"
        else:
            zero_mark = ""
        postfix_1 = f"{str(postfix_1_datetime.year)[-2:]}{zero_mark}{postfix_1_datetime.month}0{postfix_1_datetime.day}"
        current_symbol_postfix = postfix_4   
        next_symbol_postfix = postfix_1
    return current_symbol_postfix,next_symbol_postfix
#------------------------------------------------------------------------------------
class SendFile:
    """
    * 钉钉发送文件
    * 需要钉钉后台绑定ip地址https://open-dev.dingtalk.com/fe/app#/appMgr/inner/eapp/1484669569/2
    * https://open-dev.dingtalk.com/获取CorpId
    * 应用开发/企业内部开发/应用信息获取AppKey，AppSecret
    """
    #------------------------------------------------------------------------------------
    def __init__(self):
        self.appkey = "XXX"
        self.appsecret = "XXX"
    #------------------------------------------------------------------------------------
    def get_access_token(self):
        url = f"https://oapi.dingtalk.com/gettoken?appkey={self.appkey}&appsecret={self.appsecret}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"appkey": self.appkey,
                "appsecret": self.appsecret}
        data = requests.request("GET", url, data=data, headers=headers)
        access_token = data.json()["access_token"]
        return access_token
    #------------------------------------------------------------------------------------
    def get_media_id(self,file_path:str):
        access_token = self.get_access_token()  # 拿到接口凭证
        url = f"https://oapi.dingtalk.com/media/upload?access_token={access_token}&type=file"
        files = {"media": open(file_path, "rb")}
        js_data = {"access_token": access_token,
                "type": "file"}
        response = requests.post(url, files=files, data=js_data)
        data = response.json()
        if data["errcode"]:
            print(f"获取media_id出错，错误代码：{data['errcode']}，错误信息：{data['errmsg']}")
            return
        return data["media_id"]
    #------------------------------------------------------------------------------------
    def send_file(self,file_path:str):
        """
        * 发送文件到钉钉
        * 钉钉扫描http://wsdebug.dingtalk.com/定位到v0.1.2输入{"corpId":"XXX","isAllowCreateGroup":true,"filterNotOwnerGroup":false}获取chatid
        """
        access_token = self.get_access_token()
        media_id = self.get_media_id(file_path)
        chatid = "XXX"
        url = "https://oapi.dingtalk.com/chat/send?access_token=" + access_token
        header = {
            "Content-Type": "application/json"
        }
        js_data = {"access_token": access_token,
                "chatid": chatid,
                "msg": {
                    "msgtype": "file",
                    "file": {"media_id": media_id}
                }}
        request_data = requests.request("POST", url, data=json.dumps(js_data), headers=header)
        data = request_data.json()
        if data["errcode"]:
            print(f"发送文件出错，错误代码：{data['errcode']}，错误信息：{data['errmsg']}")
