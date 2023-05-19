from copy import copy
from time import sleep
import csv
import json
import sys
from pathlib import Path
from typing import (Callable, Dict,List, Union,Any,Tuple)
import logging
import secrets
from datetime import datetime,timedelta,time
from empyrical import (annual_volatility)
from math import floor, ceil
from pytz import timezone,common_timezones
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import pandas as pd
import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import redis
import talib
from dingtalkchatbot.chatbot import DingtalkChatbot
import psutil
import zlib
import cloudpickle
import h5py
import platform
from filelock import FileLock

from vnpy.trader.event import EVENT_TIMER,REDIS_CLIENT,EVENT_LOG
from vnpy.event import Event,EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.object import BarData, TickData,LogData,Status
from vnpy.trader.constant import Exchange, Interval
log_formatter = logging.Formatter("[%(asctime)s] %(message)s")
TZ_INFO = timezone(SETTINGS["timezone"])
# 当前程序主目录
PARENT_PATH = str(Path(__file__).parent.parent)
# 活动单委托状态
ACTIVE_STATUSES = [Status.NOTTRADED, Status.PARTTRADED]
# redis客户端
REDIS_POOL =redis.ConnectionPool(host = SETTINGS["redis.host"],port = SETTINGS["redis.port"],password = SETTINGS["redis.password"],socket_timeout = 300,socket_keepalive = True,retry_on_timeout =True,health_check_interval = 30)
REDIS_CLIENT = redis.StrictRedis(connection_pool = REDIS_POOL)
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
def save_csv(filepath: str, data: Any):
    """
    保存数据到csv
    """
    fieldnames = list(data.__dict__.keys())
    file_exists = Path(filepath).exists()
    
    with open(filepath, "a", newline="", encoding="utf_8_sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data.__dict__)
#------------------------------------------------------------------------------------
def save_h5(filename:str,data:Union[list,tuple,set,dict],overwrite:bool=False):
    """
    * 保存hdf5数据
    * filename文件名，data要保存的数据
    * overwrite为True覆盖源文件,为False增量更新文件
    """
    contract_file_path = get_folder_path(filename)
    filepath =f"{contract_file_path}{GetFilePath.link_sign}{filename}.h5"
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
    lock_file = filepath + '.lock'
    with FileLock(lock_file):
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
    filepath =f"{contract_file_path}{GetFilePath.link_sign}{filename}.h5"

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

    lock_file = filepath + '.lock'
    with FileLock(lock_file):
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
    index_location = defaultdict(list)
    for index, value in enumerate(values):
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
    获取32位(16进制)随机字符串
    """
    return secrets.token_hex(16)
#------------------------------------------------------------------------------------
def list_of_groups(init_list:list, children_list_len:int) -> List[List[str]]:
    """
    * 等分列表
    * init_list:要切分的列表,children_list_len:每个子列表中包含的元素数量
    """
    return [init_list[i:i+children_list_len] for i in range(0, len(init_list), children_list_len)]
#----------------------------------------------------------------------
def dict_slice(origin_dict:dict, start:int, end:int) ->Dict:
    """
    1.字典切片取值
    2.origin_dict: 字典,start: 起始,end: 终点
    """
    slice_dict = {k: v for i, (k, v) in enumerate(origin_dict.items()) if start <= i < end}
    return slice_dict
#------------------------------------------------------------------------------------
def get_float_len(value: float) -> int:
    """
    获取浮点数小数点后数值长度
    """
    if "e" in str(value):
        # 将科学计数法表示的浮点数转换为小数形式
        value = np.format_float_positional(value)
    value_str = str(value)
    _, buf = value_str.split(".")
    return len(buf)
#------------------------------------------------------------------------------------
def delete_zero(value:str) -> str:
    """
    删除字符串数值类型末尾0
    """
    # 使用 rstrip() 函数删除末尾的零
    return value.rstrip("0")
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
    symbol_mark = "".join(filter(str.isalpha,convert_contract))
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
    symbol_mark = "".join(filter(str.isdigit,convert_contract))
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
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    with FileLock(lock_file):
        try:
            with open(filepath, mode="w+", encoding="UTF-8") as file:
                json.dump(data, file, sort_keys=True, indent=4, ensure_ascii=False)

        except Exception as err:
            msg = f"文件：{filename}保存数据出错，错误信息：{err}"
            print(msg)
            return
        
#------------------------------------------------------------------------------------
def load_json(filename: str) ->Dict:
    """
    读取json文件
    """
    filepath = get_file_path(filename)
    if not filepath.exists():
        save_json(filename, {})
        return {}
    
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    with FileLock(lock_file):
        try:
            with open(filepath, mode="r", encoding="UTF-8") as file:
                data = json.load(file)
        except Exception as err:
            msg = f"文件：{filename}读取数据出错，错误信息：{err}"
            print(msg)
            data = {}
        return data
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
        rounded = round(rounded,get_float_len(target))
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
    获取合约标识(CTP接口合约标识区分大小写，数字货币接口合约标识都为大写)
    * "rb2201_SHFE/CTP"合约标识为"rb"，"ZC2201_CZCE/CTP"合约标识为"ZC"，"BTCUSD_BINANCES/BINANCES"合约标识为"BTC"
    """
    symbol,exchange,gateway_name = extract_vt_symbol(vt_symbol)
    if remain_alpha(gateway_name) in ["CTP"]:
        #CTP合约标识
        symbol_mark = remain_alpha(vt_symbol)
    elif remain_alpha(gateway_name) in ["OKEX"] and symbol[-1].isdigit():
        #OKEX接口有币本位交割(BTCUSD99)和USDT交割(BTCUSDT99)，合成指数合约需要区分symbol_mark
        symbol_mark = remain_alpha(vt_symbol).upper()
    else:    
        #数字货币合约标识
        if "." in symbol:
            symbol_mark =symbol.split(".")[0]
        elif "USDT" in symbol:
            symbol_mark = remain_alpha(symbol.split("USDT")[0]).upper()
        elif "USD" in symbol:
            symbol_mark = remain_alpha(symbol.split("USD")[0]).upper()
        elif "PERP" in symbol:
            symbol_mark = remain_alpha(symbol.split("PERP")[0]).upper()
        else:
            #其他交易所合约标识
            symbol_mark = remain_alpha(vt_symbol).upper()
            
    return symbol_mark
