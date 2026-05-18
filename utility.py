"""
常用方法组件
"""
import csv
import hashlib
import pyjson5 as jsonc
import json5 as json
import secrets
import sys
from collections import defaultdict
from copy import copy
from datetime import datetime, time, timedelta,timezone
from time import sleep
from math import ceil, floor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union,Optional
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo, available_timezones

np.seterr(divide="ignore", invalid="ignore")
import redis
import cloudpickle
import h5py
import psutil
import requests
import talib
from dingtalkchatbot.chatbot import DingtalkChatbot
from filelock import FileLock
from functools import lru_cache

from vnpy.event import Event, EventEngine,MmapSubscriber
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.event import EVENT_LOG, EVENT_TIMER
from vnpy.trader.object import BarData, LogData, Status, TickData,ContractData,ACTIVE_STATUSES
from vnpy.trader.setting import RAW_SHA512, SETTINGS, SYSTEM
#----------------------------------------------------------------------------------------------------
from zstandard import ZstdCompressor,ZstdDecompressor
COMPRESSOR = ZstdCompressor(10)     # 回测增加压缩率用10，实盘用6增加存储取速度
DECOMPRESSOR = ZstdDecompressor()
#----------------------------------------------------------------------------------------------------
# redis客户端
REDIS_POOL = redis.ConnectionPool(host=SETTINGS["redis.host"],
                                    port=SETTINGS["redis.port"],
                                    password=SETTINGS["redis.password"],
                                    socket_timeout=300,
                                    socket_keepalive=True,
                                    retry_on_timeout=True,
                                    health_check_interval=30)
REDIS_CLIENT = redis.StrictRedis(connection_pool=REDIS_POOL)
# ============ 压缩器配置 ============
# 缓存压缩器实例，避免重复创建
_COMPRESSORS = {
    'fast': ZstdCompressor(level=3),      # 速度优先（< 20MB 使用）
    'balanced': ZstdCompressor(level=6),  # 平衡模式（20-100MB 使用）
    'best': ZstdCompressor(level=10)      # 压缩率优先（> 100MB 使用）
}
DECOMPRESSOR = ZstdDecompressor()

# ============ 配置常量 ============
CHUNK_THRESHOLD = 512 * 1024        # 512KB，超过此大小启用分块
CHUNK_SIZE_SMALL = 2 * 1024 * 1024  # 2MB
CHUNK_SIZE_MEDIUM = 4 * 1024 * 1024 # 4MB
CHUNK_SIZE_LARGE = 8 * 1024 * 1024  # 8MB
PIPELINE_BATCH_SIZE = 100            # Pipeline批处理大小

#TZ_INFO = ZoneInfo(SETTINGS["timezone"])
# 当前程序主目录
PARENT_PATH = Path(__file__).parent
PARENT2_PATH = Path(__file__).parent.parent
# 此处EVENT_ENGINE仅供初始化使用，实际调用的是交易子进程里面的EVENT_ENGINE
EVENT_ENGINE = EventEngine(interval = 60)
EVENT_ENGINE.start()
# ============== 预计算常量 ==============
TZ_INFO = timezone(timedelta(hours=8))
# 时间戳长度 -> (除数, 是否需要除法)
_TS_DIVISORS: dict[int, int] = {
    10: 1,           # 秒
    13: 1_000,       # 毫秒
    16: 1_000_000,   # 微秒
    19: 1_000_000_000,  # 纳秒
}

# 预定义格式模板
_FMT_DASH = "%Y-%m-%d %H:%M:%S"
_FMT_DASH_T = "%Y-%m-%dT%H:%M:%S"
_FMT_NO_DASH = "%Y%m%d %H:%M:%S"
_FMT_NO_DASH_T = "%Y%m%dT%H:%M:%S"
@lru_cache(maxsize=64)
def _detect_format(timestamp: str) -> str:
    """缓存格式检测结果"""
    has_dash = "-" in timestamp
    has_t = "T" in timestamp
    has_micro = "." in timestamp
    
    if has_dash:
        fmt = _FMT_DASH_T if has_t else _FMT_DASH
    else:
        fmt = _FMT_NO_DASH_T if has_t else _FMT_NO_DASH
    
    return f"{fmt}.%f" if has_micro else fmt

def parse_str_timestamp(timestamp: str) -> datetime:
    """解析字符串时间戳（无时区）"""
    # 快速分割，取第一部分
    plus_idx = timestamp.find("+")
    z_idx = timestamp.find("Z")
    if plus_idx != -1:
        ts = timestamp[:plus_idx].strip()
    elif z_idx != -1:
        ts = timestamp[:z_idx].strip()
    else:
        ts = timestamp.strip()
    
    fmt = _detect_format(ts)
    return datetime.strptime(ts, fmt)

def parse_numeric_timestamp(timestamp: Union[int, float]) -> datetime:
    """解析数值时间戳"""
    if isinstance(timestamp, float):
        return datetime.fromtimestamp(timestamp, tz=None)
    
    # 整数：根据长度判断单位
    length = len(str(timestamp))
    divisor = _TS_DIVISORS.get(length, 1)
    return datetime.fromtimestamp(timestamp / divisor, tz=None)

def get_local_datetime(
    timestamp: Union[str, float, int],
    hours: int = 0
) -> datetime:
    """
    将时间戳转换为本地时间
    
    Args:
        timestamp: 时间戳（字符串/浮点/整数）
        hours: UTC 偏移小时数，默认0（0时区转到东8区）
    
    Returns:
        带时区的 datetime 对象
    """
    # 类型分发
    if isinstance(timestamp, int):
        local_time = parse_numeric_timestamp(timestamp)
    elif isinstance(timestamp, float):
        local_time = parse_numeric_timestamp(timestamp)
    elif isinstance(timestamp, str):
        # 快速判断纯数字
        if timestamp.isdigit():
            local_time = parse_numeric_timestamp(int(timestamp))
        else:
            local_time = parse_str_timestamp(timestamp)
    else:
        raise TypeError(f"不支持的类型: {type(timestamp).__name__}")
    
    # 时区转换：使用 replace 避免创建新对象
    return (local_time + timedelta(hours=hours)).replace(tzinfo=TZ_INFO)
# ----------------------------------------------------------------------------------------------------
def write_log(msg: str, gateway_name: str = ""):
    """
    写入日志信息
    """
    data = LogData(msg=msg, gateway_name=gateway_name)
    event = Event(EVENT_LOG, data)
    EVENT_ENGINE.put(event)
# ----------------------------------------------------------------------------------------------------
def mmap_event_subscriber(
    event_engine: "EventEngine",
    log_exception: Callable,
    gateway_names: List[str],
    channel: str,
) -> None:
    """
    共享内存缓存目录：windows：AppData\Local\Temp\shared_memory-rs，ubuntu：/dev/shm，如有残留可手动删除
    
    订阅共享内存事件总线并将 eTick. 事件注入本地 EventEngine。

    参数
    ----
    event_engine  : 当前进程的 EventEngine 实例
    log_exception : 异常日志回调（与原 receive_redis_stream 签名兼容）
    gateway_names : 交易接口名列表（断线时重置连接状态）
    channel       : 运行文件名（"stream_" + file_name）
    """
    global EVENT_ENGINE
    EVENT_ENGINE = event_engine
    # 推导共享内存名称
    resolved_name = derive_shm_name(channel)
    # 防止重复订阅
    if event_engine.channel:
        return
    event_engine.channel = resolved_name

    def _on_disconnect():
        msg = f"[MmapSubscriber] shm='{resolved_name}' 读取异常，已重置接口连接状态"
        write_log(msg)
        
        for gw in gateway_names:
            save_connection_status(gw, False)

    write_log(f"初始化 mmap 行情订阅，shm='{resolved_name}'")

    subscriber = MmapSubscriber(
        event_engine=event_engine,
        shm_name=resolved_name,
        log_exception=log_exception,
        on_disconnect=_on_disconnect,
    )
    event_engine.attach_subscriber(subscriber)

    # 若引擎已在运行则立即启动轮询线程；否则由 engine.start() 统一拉起
    if event_engine.is_loop_running():
        subscriber.start()

    # 保持当前线程存活（轻量等待，与原 receive_redis_stream 行为一致）
    import time as _time
    while event_engine.active:
        _time.sleep(3)
# ----------------------------------------------------------------------------------------------------
def derive_shm_name(channel: str) -> str:
    """
    从 channel 推导共享内存名称，与 MmapPublisher 使用相同规则。

    channel 示例：
        "stream_portfolio_trade_binancef_main"  → publisher 自身
        "stream_portfolio_trade_binancef_sub"   → 订阅 _main 发布的 shm
    """

    if channel.endswith("_sub"):
        core = channel.removesuffix("_sub") + "_main"
        # arbitrage <-> portfolio 互订
        if "arbitrage" in core:
            core = core.replace("arbitrage", "portfolio")
        elif "portfolio" in core:
            core = core.replace("portfolio", "arbitrage")
    else:
        # _main 或其他：直接用 channel 作为 core
        core = channel if channel.endswith("_main") else (channel.rsplit("_", 1)[0] + "_main")

    return core
# ----------------------------------------------------------------------------------------------------
def system_use():
    """
    系统信息
    """
    boot_time = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
    memory_percent = psutil.virtual_memory().percent
    net_io_counters = psutil.net_io_counters()
    recv_gb = f"{net_io_counters.bytes_recv / (1024**3):.3f} GB"
    sent_gb = f"{net_io_counters.bytes_sent / (1024**3):.3f} GB"
    
    return f"【系统启动时间:{boot_time}，内存使用率:{memory_percent}%，网卡接收流量:{recv_gb}，网卡发送流量:{sent_gb}】"
# ----------------------------------------------------------------------------------------------------
class SendMessage:
    """
    * 钉钉发送信息
    * 过滤重复消息
    """
    # 信息推送webhook与密匙
    info_webhook_1: str = "xxx"
    info_webhook_2: str = "https://oapi.dingtalk.com/robot/send?access_token=xxx"
    info_secret: str = "xxx"
    # 错误推送webhook与密匙
    error_webhook_1: str = "https://oapi.dingtalk.com/robot/send?access_token=xxx"
    error_secret: str = "xxx"

    def __init__(self, webhook: str, secret: str):
        # 初始化机器人小丁
        self.xiaoding = DingtalkChatbot(webhook, secret)
        self.last_text: str = ""  # 缓存钉钉发送的信息
    # ----------------------------------------------------------------------------------------------------
    def send_text(self, text: str):
        """
        text消息，is_at_all=True @所有人
        """
        # 过滤发送相同信息
        if self.last_text == text:
            return
        try:
            self.xiaoding.send_text(msg=f"【监控时间：{datetime.now(TZ_INFO)}】" + "\n" + f"{text}")
        except Exception as err:
            write_log(f"钉钉发送消息失败，错误信息：{err}")
        self.last_text = text
    # ----------------------------------------------------------------------------------------------------
    def send_image(self, url: str):
        """
        发送网络图片
        """
        self.xiaoding.send_image(pic_url=f"{url}")
# ----------------------------------------------------------------------------------------------------
# 钉钉推送日志信息
info_webhook = SendMessage.info_webhook_1
info_secret = SendMessage.info_secret
info_monitor = SendMessage(info_webhook, info_secret)
# ----------------------------------------------------------------------------------------------------
# 钉钉推送错误信息
error_webhook = SendMessage.error_webhook_1
error_secret = SendMessage.error_secret
error_monitor = SendMessage(error_webhook, error_secret)
# ----------------------------------------------------------------------------------------------------
def get_index_vt_symbol(vt_symbol:str) -> str:
    """
    获取指数合约vt_symbol
    """
    symbol, exchange, gateway_name = extract_vt_symbol(vt_symbol)
    # vt_symbol为指数合约直接返回vt_symbol
    if symbol.endswith("INDEX"):
        return vt_symbol
    symbol_mark = get_symbol_mark(vt_symbol)
    index_vt_symbol = f"{symbol_mark}INDEX_{exchange.value}/{gateway_name}"
    return index_vt_symbol
# ----------------------------------------------------------------------------------------------------
def save_csv(filepath: str, data: Any):
    """
    保存数据到csv
    """
    fieldnames = list(data.__dict__.keys())
    file_exists = Path(filepath).exists()
    lock_file = filepath + ".lock"
    with FileLock(lock_file):
        with open(filepath, "a", newline="", encoding="utf_8_sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data.__dict__)
# ----------------------------------------------------------------------------------------------------
def save_h5(filename: str, data: Union[list, tuple, set, dict], overwrite: bool = False):
    """
    * 保存hdf5数据
    * filename文件名，data要保存的数据
    * overwrite为True覆盖源文件,为False增量更新文件
    """
    contract_file_path = get_folder_path(filename)
    filepath = contract_file_path / f"{filename}.h5"
    if overwrite:
        raw_data = data
    else:
        # 只在读取到原始数据后增量更新数据，避免空字典数据写入，原始数据为空直接写入新数据
        raw_data: Union[list, tuple, set, dict] = load_h5(filename)
        if raw_data:
            if isinstance(raw_data, dict) and isinstance(data, dict):
                raw_data.update(data)
            elif isinstance(raw_data, tuple) and isinstance(data, tuple):
                raw_data = (*raw_data, *data)
            elif isinstance(raw_data, set) and isinstance(data, set):
                raw_data = raw_data | data
            elif isinstance(raw_data, list) and isinstance(data, list):
                raw_data = list_de_duplication(raw_data + data)
        else:
            raw_data = data
    # 循环写入h5数据直到写入成功或重试3次后退出循环
    count = 0
    while True:
        count += 1
        status = save_h5_status(filepath, raw_data)
        if status or count > 3:
            break
# ----------------------------------------------------------------------------------------------------
def save_h5_status(filepath: Path, raw_data: Any):
    """
    获取H5保存数据状态
    """
    filepath = str(filepath)
    lock_file = filepath + ".lock"
    with FileLock(lock_file):
        try:
            with h5py.File(filepath, "w") as file:
                data = COMPRESSOR.compress(cloudpickle.dumps(raw_data))
                file.create_dataset("data", data=np.frombuffer(data, dtype=np.uint8))
            return True
        except Exception as err:
            write_log(f"保存H5文件：{filepath}出错，错误信息：{err}，要保存的数据：{raw_data}")
            return False
# ----------------------------------------------------------------------------------------------------
def load_h5(filename: str):
    """
    读取hdf5数据
    """
    contract_file_path = get_folder_path(filename)
    filepath = contract_file_path / f"{filename}.h5"

    if not Path(filepath).exists():
        return {}
    count = 0
    while True:
        count += 1
        status, data = load_h5_status(filepath)
        if status or count > 3:
            return data
# ----------------------------------------------------------------------------------------------------
def load_h5_status(filepath: Path):
    """
    获取H5读取状态及数据
    """
    filepath = str(filepath)
    lock_file = filepath + ".lock"
    with FileLock(lock_file):
        try:
            with h5py.File(filepath, "r") as file:
                data = file["data"][()].tobytes()       # [()] 对标量和数组都适用，写法更通用
                data = cloudpickle.loads(DECOMPRESSOR.decompress(data),encoding="UTF-8")
                return True, data
        except Exception as err:
            write_log(f"读取H5文件：{filepath}出错，错误信息：{err}")
            return False, {}
# ----------------------------------------------------------------------------------------------------
def index_location(values: list):
    """
    获取列表相同值索引
    """
    index_location = defaultdict(list)
    for index, value in enumerate(values):
        index_location[value].append(index)
    return index_location
# ----------------------------------------------------------------------------------------------------
def list_de_duplication(value: List):
    """
    列表去重复且保持原顺序
    """
    return sorted(set(value), key=value.index)
# ----------------------------------------------------------------------------------------------------
def get_uuid():
    """
    获取32位(16进制)随机字符串
    """
    return secrets.token_hex(16)
# ----------------------------------------------------------------------------------------------------
def list_of_groups(init_list: list, children_list_len: int) -> List[List[str]]:
    """
    * 等分列表
    * init_list:要切分的列表,children_list_len:每个子列表中包含的元素数量
    """
    return [init_list[i : i + children_list_len] for i in range(0, len(init_list), children_list_len)]
# ----------------------------------------------------------------------
def dict_slice(origin_dict: dict, start: int, end: int) -> Dict:
    """
    1.字典切片取值
    2.origin_dict: 字典,start: 起始,end: 终点
    """
    slice_dict = {k: v for i, (k, v) in enumerate(origin_dict.items()) if start <= i < end}
    return slice_dict
# ----------------------------------------------------------------------
def is_target_contract(vt_symbol:str, target_gateway_name:str) -> bool:
    """
    * 判断合约是否为目标交易接口的合约，排除指数(symbol以INDEX结尾)合约
    """
    symbol, exchange, gateway_name = extract_vt_symbol(vt_symbol)
    return gateway_name == target_gateway_name and not symbol.endswith("INDEX")
# ----------------------------------------------------------------------------------------------------
def remain_alpha(convert_contract: str) -> str:
    """
    返回合约symbol或字符串的字母部分
    """
    if "_" in convert_contract and "/" in convert_contract:
        convert_contract = extract_vt_symbol(convert_contract)[0]
    else:
        if "_" in convert_contract:
            convert_contract = convert_contract.split("_")[0]
    symbol_mark = "".join(filter(str.isalpha, convert_contract))
    return symbol_mark
# ----------------------------------------------------------------------------------------------------
def remain_digit(convert_contract: str) -> str:
    """
    返回合约symbol或字符串的数字部分
    """
    if "_" in convert_contract and "/" in convert_contract:
        convert_contract = extract_vt_symbol(convert_contract)[0]
    else:
        if "_" in convert_contract:
            convert_contract = convert_contract.split("_")[0]
    symbol_mark = "".join(filter(str.isdigit, convert_contract))
    return symbol_mark
# ----------------------------------------------------------------------------------------------------
def remain_alpha_numeric(convert_contract: str) -> str:
    """
    返回合约symbol或字符串的字母和数字部分
    """
    if "_" in convert_contract and "/" in convert_contract:
        convert_contract = extract_vt_symbol(convert_contract)[0]
    else:
        if "_" in convert_contract:
            convert_contract = convert_contract.split("_")[0]
    symbol_mark = "".join(filter(lambda x: x.isalpha() or x.isdigit(), convert_contract))
    return symbol_mark
# ----------------------------------------------------------------------------------------------------
def str_to_number(value:str):
    """
    * 转换字符串为数字，如果转换失败返回0
    """
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return 0
# ----------------------------------------------------------------------------------------------------
def is_usdt_contract(vt_symbol:str):
    """
    判断合约是否是USDT合约
    """
    symbol,exchange,gateway_name = extract_vt_symbol(vt_symbol)
    return ("USDT" in symbol)
# ----------------------------------------------------------------------------------------------------
def is_usdc_contract(vt_symbol:str):
    """
    判断合约是否是USDC合约
    """
    symbol,exchange,gateway_name = extract_vt_symbol(vt_symbol)

    # hyperliquid,dydx交易所USDC合约由exchange判定
    if exchange == Exchange.DYDX:
        return True
    if exchange == Exchange.HYPE:
        if symbol.startswith(("km:","flx:","vntl:")):
            return False
        else:
            return True
    # 过滤反向合约交易所
    if exchange in [Exchange.BINANCEF]:
        return False
    return ("USDC" in symbol or "PERP" in symbol)
# ----------------------------------------------------------------------------------------------------
def is_usdh_contract(vt_symbol:str):
    """
    判断合约是否是USDH合约
    """
    symbol,*_ = extract_vt_symbol(vt_symbol)
    return symbol.startswith(("km:","flx:","vntl:"))
# ----------------------------------------------------------------------------------------------------
def extract_vt_symbol(vt_symbol: str) -> Tuple[str, Exchange, str]:
    """
    返回(symbol:str, exchange: Exchange,gateway_name:str)
    """
    symbol_exchange, gateway_name = vt_symbol.rsplit("/",1)
    symbol, exchange = symbol_exchange.rsplit("_", 1)
    return symbol, Exchange(exchange), gateway_name
# ----------------------------------------------------------------------------------------------------
def save_connection_status(gateway_name: str, status: bool, msg: str = ""):
    """
    * 保存交易接口连接状态
    * 参数说明:
        gateway_name: 交易接口的名称。
        status: 交易接口的连接状态，True 表示已连接，False 表示未连接。
        msg: 可选参数，用于记录与当前操作相关的任何日志信息。
    """
    # gateway_name为空值直接返回
    if not gateway_name:
        return
    if msg:
        write_log(msg)
    connection_status = load_json("connection_status.json")
    connection_status.update({gateway_name: status})
    save_json("connection_status.json", connection_status)
# ----------------------------------------------------------------------------------------------------
def _select_compressor(data_size: int) -> ZstdCompressor:
    """
    根据数据大小选择最优压缩策略
    
    策略说明：
    - < 20MB: 速度优先 (level 3) - 小数据快速处理
    - 20-100MB: 平衡模式 (level 6) - 兼顾速度和压缩率
    - > 100MB: 压缩率优先 (level 10) - 大数据节省存储空间
    
    Args:
        data_size: 原始数据大小（字节）
    
    Returns:
        对应的压缩器实例
    """
    if data_size < 20 * 1024 * 1024:       # < 20MB: 速度优先
        return _COMPRESSORS['fast']
    elif data_size < 100 * 1024 * 1024:    # 20-100MB: 平衡
        return _COMPRESSORS['balanced']
    else:                                   # > 100MB: 压缩率优先
        return _COMPRESSORS['best']

def _get_chunk_size(data_size: int) -> int:
    """
    根据压缩后数据大小选择最优分块大小
    
    Args:
        data_size: 压缩后数据大小（字节）
    
    Returns:
        分块大小（字节）
    """
    if data_size < 10 * 1024 * 1024:
        return CHUNK_SIZE_SMALL
    elif data_size < 100 * 1024 * 1024:
        return CHUNK_SIZE_MEDIUM
    return CHUNK_SIZE_LARGE

def save_redis_data(
    file_name: str,
    data: Any,
    expire_minute: int = 1440,
    use_lock: bool = False
) -> bool:
    """
    优化版 Redis 数据存储
    
    特性：
    1. 智能压缩：根据数据大小自动选择压缩级别
    2. 分块存储：大数据自动分块，避免单个 key 过大
    3. 内存优化：使用 memoryview 避免数据拷贝
    4. 批量操作：Pipeline 批处理提升性能
    
    Args:
        file_name: 存储键名
        data: 要存储的数据对象
        expire_minute: 过期时间（分钟），默认 1440（24小时）
        use_lock: 是否使用分布式锁，默认 False
    
    Returns:
        是否存储成功
    """
    lock: Optional[redis.lock.Lock] = None
    
    try:
        # 获取分布式锁
        if use_lock:
            lock = REDIS_CLIENT.lock(f"lock:{file_name}", blocking_timeout=10)
            lock.acquire(blocking=True)
        
        # 序列化数据
        serialized = cloudpickle.dumps(data)
        original_size = len(serialized)
        
        # 根据数据大小选择压缩策略
        compressor = _select_compressor(original_size)
        compressed = compressor.compress(serialized)
        compressed_size = len(compressed)
        
        # 释放原始数据，节省内存
        del serialized
        
        # 计算过期时间
        expire_time = datetime.now() + timedelta(minutes=expire_minute)
        
        # 根据压缩后大小选择存储模式
        if compressed_size <= CHUNK_THRESHOLD:
            # 简单模式：数据较小，直接存储
            REDIS_CLIENT.set(file_name, compressed)
            REDIS_CLIENT.expireat(file_name, expire_time)
        else:
            # 分块模式：数据较大，分块存储
            chunk_size = _get_chunk_size(compressed_size)
            chunk_count = (compressed_size + chunk_size - 1) // chunk_size
            
            # 使用 memoryview 避免切片时的内存拷贝
            mv = memoryview(compressed)
            pipe = REDIS_CLIENT.pipeline(transaction=False)
            
            # 分块存储
            for i in range(chunk_count):
                start = i * chunk_size
                end = min(start + chunk_size, compressed_size)
                chunk_key = f"{file_name}:chunk:{i}"
                
                pipe.set(chunk_key, bytes(mv[start:end]))
                pipe.expireat(chunk_key, expire_time)
                
                # 批量执行，避免 pipeline 过大导致内存问题
                if (i + 1) % PIPELINE_BATCH_SIZE == 0:
                    pipe.execute()
                    pipe = REDIS_CLIENT.pipeline(transaction=False)
            
            # 执行剩余操作
            pipe.execute()
            
            # 存储元数据（JSON 格式，比 Hash 更省内存）
            metadata = json.dumps({
                "c": chunk_count,       # chunk_count
                "s": chunk_size,        # chunk_size  
                "z": compressed_size,   # compressed_size
                "o": original_size,     # original_size
                "t": datetime.now().timestamp()  # timestamp
            })
            
            metadata_key = f"{file_name}:meta"
            REDIS_CLIENT.set(metadata_key, metadata)
            REDIS_CLIENT.expireat(metadata_key, expire_time)
        
        return True
        
    except Exception as e:
        write_log(f"存储失败 ({file_name}): {e}")
        return False
        
    finally:
        # 释放锁
        if use_lock and lock and lock.locked():
            lock.release()

def load_redis_data(file_name: str, use_lock: bool = False) -> Any:
    """
    优化版 Redis 数据加载
    
    特性：
    1. 自动识别存储模式（简单/分块）
    2. MGET 批量获取：单次网络往返获取所有分块
    3. 异常处理：分块丢失自动检测
    
    Args:
        file_name: 存储键名
        use_lock: 是否使用分布式锁，默认 False
    
    Returns:
        反序列化后的数据对象，失败返回空字典
    """
    lock: Optional[redis.lock.Lock] = None
    
    try:
        # 获取分布式锁
        if use_lock:
            lock = REDIS_CLIENT.lock(f"lock:{file_name}", blocking_timeout=10)
            lock.acquire(blocking=True)
        
        # 检查元数据，判断存储模式
        metadata_key = f"{file_name}:meta"
        meta_raw = REDIS_CLIENT.get(metadata_key)
        
        if meta_raw:
            # 分块模式
            meta = json.loads(meta_raw)
            chunk_count = meta["c"]
            
            # 使用 MGET 批量获取所有分块（单次网络往返）
            chunk_keys = [f"{file_name}:chunk:{i}" for i in range(chunk_count)]
            chunks = REDIS_CLIENT.mget(chunk_keys)
            
            # 检查分块完整性
            if not all(chunks):
                missing = [i for i, c in enumerate(chunks) if c is None]
                raise ValueError(f"分块丢失，索引: {missing}")
            
            # 合并分块
            compressed = b''.join(chunks)
        else:
            # 简单模式
            compressed = REDIS_CLIENT.get(file_name)
            if not compressed:
                return {}
        
        # 解压并反序列化
        decompressed = DECOMPRESSOR.decompress(compressed)
        return cloudpickle.loads(decompressed, encoding="UTF-8")
        
    except Exception as e:
        write_log(f"加载失败 ({file_name}): {e}")
        return {}
        
    finally:
        # 释放锁
        if use_lock and lock and lock.locked():
            lock.release()

def delete_redis_data(file_name: str, use_lock: bool = False) -> bool:
    """
    删除 Redis 数据
    
    自动识别存储模式并删除所有相关键
    
    Args:
        file_name: 存储键名
        use_lock: 是否使用分布式锁，默认 False
    
    Returns:
        是否删除成功
    """
    lock: Optional[redis.lock.Lock] = None
    
    try:
        # 获取分布式锁
        if use_lock:
            lock = REDIS_CLIENT.lock(f"lock:{file_name}", blocking_timeout=10)
            lock.acquire(blocking=True)
        
        # 检查存储模式
        metadata_key = f"{file_name}:meta"
        meta_raw = REDIS_CLIENT.get(metadata_key)
        
        if meta_raw:
            # 分块模式：删除所有分块和元数据
            meta = json.loads(meta_raw)
            chunk_count = meta["c"]
            
            # 构建所有相关键
            keys_to_delete = [f"{file_name}:chunk:{i}" for i in range(chunk_count)]
            keys_to_delete.append(metadata_key)
            
            # 批量删除
            REDIS_CLIENT.delete(*keys_to_delete)
        else:
            # 简单模式：直接删除
            REDIS_CLIENT.delete(file_name)
        
        return True
        
    except Exception as e:
        write_log(f"删除失败 ({file_name}): {e}")
        return False
        
    finally:
        # 释放锁
        if use_lock and lock and lock.locked():
            lock.release()

def get_redis_data_info(file_name: str) -> dict:
    """
    获取 Redis 数据信息
    
    Args:
        file_name: 存储键名
    
    Returns:
        数据信息字典，包含存在状态、存储模式、大小、压缩率等
    """
    try:
        # 检查分块模式
        metadata_key = f"{file_name}:meta"
        meta_raw = REDIS_CLIENT.get(metadata_key)
        
        if meta_raw:
            meta = json.loads(meta_raw)
            compression_ratio = meta["o"] / meta["z"] if meta["z"] > 0 else 0
            
            return {
                "exists": True,
                "mode": "chunked",
                "chunk_count": meta["c"],
                "chunk_size": meta["s"],
                "compressed_size": meta["z"],
                "original_size": meta["o"],
                "compression_ratio": round(compression_ratio, 2),
                "saved_bytes": meta["o"] - meta["z"],
                "timestamp": meta["t"]
            }
        
        # 检查简单模式
        data = REDIS_CLIENT.get(file_name)
        if data:
            return {
                "exists": True,
                "mode": "simple",
                "compressed_size": len(data)
            }
        
        return {"exists": False}
        
    except Exception as e:
        write_log(f"获取信息失败 ({file_name}): {e}")
        return {"exists": False, "error": str(e)}
# ----------------------------------------------------------------------------------------------------
def generate_vt_symbol(symbol: str, exchange: Exchange, gateway_name: str):
    """
    生成vt_symbol
    """
    return f"{symbol}_{exchange.value}/{gateway_name}"
# ----------------------------------------------------------------------------------------------------
def get_trader_dir(temp_name: str):
    """
    获取 .vntrader 工作路径。如果当前工作路径下不存在该目录，则使用系统主目录，
    并确保该目录存在。
    """
    cwd = Path.cwd()
    home_path = Path.home()

    # 尝试在当前工作路径下查找 .vntrader
    temp_path = cwd.joinpath(temp_name)
    if temp_path.exists():
        return cwd, temp_path

    # 如果当前工作路径下没有，尝试在用户主目录下查找
    temp_path = home_path.joinpath(temp_name)
    temp_path.mkdir(exist_ok=True)  # 确保目录存在，不存在则创建

    return home_path, temp_path

TRADER_DIR, TEMP_DIR = get_trader_dir(".vntrader")
sys.path.append(str(TRADER_DIR))
# ----------------------------------------------------------------------------------------------------
def get_file_path(filename: str):
    """
    返回文件路径
    """
    return TEMP_DIR.joinpath(filename)
# ----------------------------------------------------------------------------------------------------
def get_folder_path(folder_name: str) -> Path:
    """
    返回文件夹路径
    """
    folder_path = TEMP_DIR.joinpath(folder_name)
    if not folder_path.exists():
        folder_path.mkdir()
    return folder_path
# ----------------------------------------------------------------------------------------------------
def get_icon_path(filepath: str, ico_name: str):
    """
    返回图标路径
    """
    ui_path = Path(filepath).parent
    icon_path = ui_path.joinpath("ico", ico_name)
    return str(icon_path)
# ----------------------------------------------------------------------------------------------------
def save_json(filename: str, data: Union[List, Dict]):
    """
    保存数据到json文件
    """
    # 监控文件名
    monitor_names = ["data_recorder_setting.json", "connection_status.json", "html_send_status.json"]
    # 监控文件保存数据为空发送错误信息到钉钉
    if not data and filename in monitor_names:
        error_monitor.send_text(f"文件：{filename}要保存的数据为空，请立即核实程序运行状况")
        return

    filepath = get_file_path(filename)
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    with FileLock(lock_file):
        try:
            with open(filepath, mode="w", encoding="UTF-8") as file:
                json.dump(data, file, sort_keys=True, indent=4, ensure_ascii=False,allow_duplicate_keys=False)
                #pyjson5写入数据
                #jsonc.encode_io(data,file,supply_bytes=False)
        except Exception as err:
            msg = f"文件：{filename}保存数据出错，错误信息：{err}"
            write_log(msg)
            if filename in monitor_names:
                error_monitor.send_text(msg)
            return
# ----------------------------------------------------------------------------------------------------
def load_json(filename: str) -> Dict:
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
                # json5读取数据太慢使用pyjson5读取
                #data = json.load(file,allow_duplicate_keys=False)
                data = jsonc.decode_io(file)
        except Exception as err:
            msg = f"文件：{filename}读取数据出错，错误信息：{err}"
            write_log(msg)
            data = {}
        return data
# ----------------------------------------------------------------------------------------------------
# 启动程序时缓存dr_data
DR_DATA = load_json("data_recorder_setting.json")
# ----------------------------------------------------------------------------------------------------
def count_decimal_places(number: Union[int, float]) -> int:
    """
    计算浮点数小数点后的位数。适用于整数、浮点数及科学计数法表示的数。
    """
    # 如果数值是整数，则小数部分位数为0
    if number.is_integer():
        return 0
    if "e" in str(number) or "E" in str(number):
        # 将科学计数法表示的浮点数转换为小数形式
        number = np.format_float_positional(number,trim = "-")      # trim -,保留原始格式
    _, buf = str(number).split(".")
    return len(buf)
# ----------------------------------------------------------------------------------------------------
def round_to(value: float, target: Union[int, float]) -> Union[int, float]:
    """
    将给定的数值 `value` 按照 `target` 的最小变动单位进行取整。

    参数:
        value (float): 要进行取整的数值。
        target (Union[int, float]): 最小变动单位，可以是整数或浮点数。

    返回:
        Union[int, float]: 取整后的数值。如果 `target` 为整数，返回整数；
                        如果 `target` 为浮点数，返回浮点数。
    """
    value = np.nan_to_num(value)
    try:
        rounded = int(round(value / target)) * target
    except Exception:
        rounded = 0
    # 浮点数再次取整,防止返回数值精度不对
    if isinstance(rounded, float):
        rounded = round(rounded, count_decimal_places(target))
    return rounded
# ----------------------------------------------------------------------------------------------------
def floor_to(value: Union[str, float], target: float) -> float:
    """
    将给定数值向下取整到最接近的目标值的倍数。
    
    参数:
        value (Union[str, float]): 需要进行向下取整操作的数值，可以是字符串或浮点数类型。如果传入的是字符串，则会尝试将其转换为浮点数。
        target (float): 目标值，表示需要将value向下取整到最接近该目标值的倍数。例如，若target=0.1，则value会被向下取整到最接近0.1的倍数（如3.45会变成3.4）。
        
    返回:
        float: 向下取整后的结果，返回一个浮点数值。
    """
    tmp = float(target)
    rounded = int(floor(np.nan_to_num(float(value)) / tmp)) * tmp
    return rounded
# ----------------------------------------------------------------------------------------------------
def ceil_to(value: Union[str, float], target: float) -> float:
    """
    将给定的数值向上取整到最接近的目标变动单位。
    
    这个函数用于将一个数值（可以是字符串或浮点数）向上调整为最接近的目标变动单位的倍数。
    例如，如果目标变动单位是0.5，那么传入1.2的值将会被调整为1.5。

    参数:
        value (Union[str, float]): 需要进行向上取整操作的数值。可以是一个字符串或浮点数。
        target (float): 目标变动单位，即数值应当被向上取整到的最小变动步长。必须是正数。

    返回值:
        float: 向上取整后的结果，返回一个浮点数。
    """
    tmp = float(target)
    rounded = int(ceil(np.nan_to_num(float(value)) / tmp)) * tmp
    return rounded
# ----------------------------------------------------------------------------------------------------
def add_timezone(dt):
    """
    datetime添加时区,用isinstance判定实例会堵塞,只能用try except
    """
    if dt.tzinfo:
        return dt
    dt = dt.replace(tzinfo = TZ_INFO)
    return dt
# ----------------------------------------------------------------------------------------------------
def get_symbol_mark(vt_symbol: str) -> str:
    """
    获取合约标识(CTP接口合约标识区分大小写，数字货币接口合约标识都为大写)
    * "rb2201_SHFE/CTP"合约标识为"rb"
    * "ZC2201_CZCE/CTP"合约标识为"ZC"
    * "BTCUSD_BINANCES/BINANCES"合约标识为"BTC"
    """
    symbol, exchange, gateway_name = extract_vt_symbol(vt_symbol)
    gateway_name_alpha = remain_alpha(gateway_name)
    # 指数合约标识为symbol去除INDEX
    index_suffix = "INDEX"
    if symbol.endswith(index_suffix):
        return symbol[:-len(index_suffix)]
    # CTP接口合约标识只保留字母部分
    if gateway_name_alpha == "CTP":
        return remain_alpha(vt_symbol)
    # 数字货币接口合约标识保留结算货币分割前的字母和数字部分
    # OKX接口有币本位交割(BTCUSD240927和USDT交割(BTCUSDT240927)，合成指数合约需要区分symbol_mark
    if gateway_name_alpha == "OKX":
        if symbol[-6:].isdigit():
            return remain_alpha(symbol).upper()
    # BYBITONE接口有币本位交割(BTCUSDZ25)和USDT交割(BTCUSDT-11APR25)，合成指数合约需要区分symbol_mark
    elif gateway_name_alpha == "BYBITONE":
        if symbol[-2:].isdigit() and not symbol.endswith("1000"):
            if "USDT" in symbol:
                symbol_mark = symbol.split("USDT")[0] + "USDT"
                return symbol_mark
            elif "USD" in symbol:
                symbol_mark = symbol.split("USD")[0] + "USD"
                return symbol_mark
    currency_identifiers = ["USD", "PERP"]      #USD分割包括USDT,USDC
    for currency in currency_identifiers:
        if currency in symbol:
            return remain_alpha_numeric(symbol.split(currency)[0]).upper()
    # 其他交易所合约标识只保留字母部分
    return remain_alpha(vt_symbol).upper()
# ----------------------------------------------------------------------------------------------------
def virtual(func: "callable"):
    """
    创建"virtual"装饰器，需要在子类实现的方法使用该装饰器
    """
    return func
# ----------------------------------------------------------------------------------------------------
def quarter_date_count(count_datetime: datetime):
    """
    计算季度合约的目标日期
    
    该函数根据给定的日期，计算出对应的季度合约交割日期(季度最后一个星期五)。
    
    参数:
        count_datetime: datetime, 输入的日期时间对象，用于计算季度合约的目标日期
    
    返回:
        year: int, 目标日期的年份
        month: int, 目标日期的月份
        day: int, 目标日期的具体日期
    """
    # 获取计算日期月份
    count_month = count_datetime.month

    # 根据月份选择目标月份和目标日期
    target_dates = {
        (1, 3): (3, f"{count_datetime.year}-04-01"),
        (4, 6): (6, f"{count_datetime.year}-07-01"),
        (7, 9): (9, f"{count_datetime.year}-10-01"),
        (10, 12): (12, f"{count_datetime.year+1}-01-01")
    }

    # 获取目标月份和目标日期
    target_month, target_date_str = next(value for key, value in target_dates.items() if key[0] <= count_month <= key[1])
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    # 计算季度合约交割日
    days_ago = (7 + target_date.weekday() - 4) % 7
    target_date -= timedelta(days=days_ago if days_ago != 0 else 7)
    
    # 返回目标年份、月份、日期
    return target_date.year, target_month, target_date.day
# ----------------------------------------------------------------------------------------------------
def get_quarter_postfix(gateway_name: str, count_datetime: datetime = None):
    """
    返回季度合约symbol后缀
    
    参数:
        gateway_name (str): 交易所名称，用于判断结算规则，默认值为None
        count_datetime (datetime): 当前时间，如果未提供，则使用当前时区的时间

    返回:
        symbol_postfix (str): 季度合约的symbol后缀，格式为yyMMdd（两位年份、两位月份和两位日期）
    
    注意:
        - 该函数主要用于期货或期权等金融工具的季度合约命名
        - 不同交易所可能有不同的结算规则，如火必(HUOBIF)和欧易(OKX)
    """
    if not count_datetime:
        count_datetime = datetime.now(TZ_INFO)

    quarter_year, quarter_month, quarter_day = quarter_date_count(count_datetime)
    # 季度合约到达换月日期后计算下个季度合约
    # 火必,欧易在季度第二周周五结算，其他数字货币交易所在季度合约当天结算
    if gateway_name in ["HUOBIF", "OKX"]:
        next_quarter_count = count_datetime.day >= quarter_day - 14
    else:
        next_quarter_count = count_datetime.day >= quarter_day
    if quarter_year == count_datetime.year and quarter_month == count_datetime.month and next_quarter_count and count_datetime.hour >= 16:
        target_datetime = count_datetime + relativedelta(months=1)
        quarter_year, quarter_month, quarter_day = quarter_date_count(target_datetime)

    symbol_postfix = f"{quarter_year % 100:02}{quarter_month:02}{quarter_day:02}"
    return symbol_postfix
# -----------------------------------------------------------------------
def current_date_count(count_month: int):
    """
    计算给定月份的最后一个周五所在日期，并返回该日期对应的年份、月份和具体日期。
    参数：
        count_month (int): 当前月份，范围为1到12，默认无默认值。
    返回值：
        tuple: 包含三个元素的元组，分别对应于年份(int)、月份(int)以及具体的日数(int)。
    """
    end_date = datetime.now()
    year = end_date.year

    if count_month < 12:
        target_date = datetime.strptime(f"{year}-{count_month + 1}-01", "%Y-%m-%d")
    else:
        target_date = datetime.strptime(f"{year + 1}-01-01", "%Y-%m-%d")

    days_ago = (7 + target_date.weekday() - 4) % 7  # 周五结算,weekday:4
    target_date -= timedelta(days=(days_ago if days_ago != 0 else 7))

    return target_date.year, count_month, target_date.day
# -----------------------------------------------------------------------
def get_friday_postfix():
    """
    返回月份每个周五所在日期str
    """
    now = datetime.now()
    current_month = now.month

    target_year, target_month, target_day = current_date_count(current_month)

    if now.day > target_day:
        target_year, target_month, target_day = current_date_count(current_month + 1)
    # target_year % 100：计算表示年份的整数与 100 的余数，从而提取年份的最后两位数字
    year_2_digit = str(target_year % 100).zfill(2)
    month_2_digit = str(target_month).zfill(2)
    postfix_1 = f"{year_2_digit}{month_2_digit}{max(1, target_day - 21):02}"
    postfix_2 = f"{year_2_digit}{month_2_digit}{max(1, target_day - 14):02}"
    postfix_3 = f"{year_2_digit}{month_2_digit}{max(1, target_day - 7):02}"
    postfix_4 = f"{year_2_digit}{month_2_digit}{target_day:02}"

    return postfix_1, postfix_2, postfix_3, postfix_4
# -----------------------------------------------------------------------
def get_current_next_postfix():
    """
    返回当周，次周合约后缀
    """
    postfix_1, postfix_2, postfix_3, postfix_4 = get_friday_postfix()
    now = datetime.now()
    int_now_date = int(f"{now.month:02}{now.day:02}")

    if int_now_date <= int(postfix_1[-4:]):
        current_symbol_postfix, next_symbol_postfix = postfix_1, postfix_2
    elif int(postfix_1[-4:]) < int_now_date <= int(postfix_2[-4:]):
        current_symbol_postfix, next_symbol_postfix = postfix_2, postfix_3
    elif int(postfix_2[-4:]) < int_now_date <= int(postfix_3[-4:]):
        current_symbol_postfix, next_symbol_postfix = postfix_3, postfix_4
    elif int(postfix_3[-4:]) < int_now_date <= int(postfix_4[-4:]):
        postfix_1_datetime = datetime.strptime(f"{now.year}{postfix_4[-4:]}", "%Y%m%d") + timedelta(days=7)
        postfix_1 = f"{postfix_1_datetime.year % 100:02}{postfix_1_datetime.month:02}{postfix_1_datetime.day:02}"
        current_symbol_postfix, next_symbol_postfix = postfix_4, postfix_1

    return current_symbol_postfix, next_symbol_postfix
# --------------------------------------------------------------------------------
def get_hex(file_path: Path):
    """
    获取文件的SHA512哈希值
    """
    with open(file_path, "rb") as file:
        hex_value = hashlib.sha512(file.read()).hexdigest()
        write_log(f"文件：{file_path}的SHA512哈希值为：\n{hex_value}")
        return hex_value
# --------------------------------------------------------------------------------
class GetFilePath:
    """
    * 获取文件路径
    * 该类必须放在init函数里面执行或者全局变量避免内存泄露
    """
    # 从data_recorder_setting读取所有合约列表
    dr_data = DR_DATA
    recording_list: List[str] = dr_data["recording_list"]  # 从data_recorder_setting获取所有活跃标的
    index_list: List[str] = [vt_symbol for vt_symbol in recording_list if extract_vt_symbol(vt_symbol)[0].endswith("INDEX")]  # 指数合约列表
    # cta_strategy设置
    cta_strategy_setting = load_json("cta_strategy_setting.json")
    # cta_strategy参数
    cta_strategy_data = load_json("cta_strategy_data.json")
    # portfolio_strategy设置
    portfolio_strategy_setting = load_json("portfolio_strategy_setting.json")
    # portfolio_strategy参数
    portfolio_strategy_data = load_json("portfolio_strategy_data.json")
    # 价差合约参数
    spread_params_setting = load_json("spread_params_setting.json")
    # 所有交易合约
    cta_vt_symbols: List[str] = list_de_duplication([data["vt_symbol"] for data in cta_strategy_setting.values()])
    portfolio_vt_symbols: List[List[str]] = [data["vt_symbols"] for data in portfolio_strategy_setting.values()]
    portfolio_vt_symbols: List[str] = list_de_duplication([vt_symbol for vt_symbols in portfolio_vt_symbols for vt_symbol in vt_symbols])
    spread_vt_symbols: List[str] = list_de_duplication([data["vt_symbol"] for raw in spread_params_setting for data in raw["leg_settings"]])
    all_trading_vt_symbols: List[str] = cta_vt_symbols + portfolio_vt_symbols + spread_vt_symbols
    # 路径连接符和选择模型训练设备
    if SYSTEM == "Windows":
        device = "gpu"
    elif SYSTEM == "Linux":
        device = "cpu"
    # data_recorder路径
    dr_data_path: Path = TEMP_DIR / "data_recorder_setting.json"
    # cta_strategy策略设置，参数路径
    cta_strategy_setting_path: Path = TEMP_DIR / "cta_strategy_setting.json"
    cta_strategy_data_path: Path = TEMP_DIR / "cta_strategy_data.json"
    # portfolio_strategy策略设置，参数路径
    portfolio_strategy_setting_path: Path = TEMP_DIR / "portfolio_strategy_setting.json"
    portfolio_strategy_data_path: Path = TEMP_DIR / "portfolio_strategy_data.json"
    # postgres数据路径
    postgres_data_path: Path = PARENT2_PATH / "app" / "portfolio_strategy" / "postgres_csv数据"
    # 账户数据父目录
    account_parent_path: Path = PARENT2_PATH / "app" / "portfolio_strategy" / "account_info"
    # ----------------------------------------------------------------------------------------------------
    def __init__(self):
        self.reboot_gateway_name: str = ""
        self.reboot_datetime: Optional[datetime] = None  # 删除过期合约时间缓存
        self.reboot_timer_started: bool = False  # 定时重启事件启动状态
        self.update_datetime()
        EVENT_ENGINE.register(EVENT_TIMER, self.process_timer_event)
    # ----------------------------------------------------------------------------------------------------
    def update_datetime(self):
        self.now_date = datetime.now(TZ_INFO).strftime("%Y-%m-%d")  # 当前年月日字符串
        self.now_datetime = datetime.now(TZ_INFO).strftime("%Y-%m-%d %H-%M-%S")  # 当前日期字符串，精确到秒
    # ----------------------------------------------------------------------------------------------------
    def process_timer_event(self, event: Event):
        """
        定时更新日期
        """
        self.update_datetime()
    # ----------------------------------------------------------------------------------------------------
    def delete_dr_data(self, delete_symbol: str, delete_gateway_name: str, restart: bool = True):
        """
        * 从dr_data bar字典和recording_list中删除合约数据
        * restart 为True定时重启交易子进程
        """
        for vt_symbol in self.recording_list[:]:  # 注意：用切片复制列表以避免迭代时修改
            symbol, exchange, gateway_name = extract_vt_symbol(vt_symbol)
            if symbol == delete_symbol and gateway_name == delete_gateway_name:
                self.recording_list.remove(vt_symbol)
                self.dr_data["bar"].pop(vt_symbol, None)  # 使用 pop 的默认值避免 KeyError

        save_json("data_recorder_setting.json", self.dr_data)

        if restart and not self.reboot_timer_started:
            self.reboot_gateway_name = delete_gateway_name
            self.reboot_datetime = datetime.now(TZ_INFO)
            EVENT_ENGINE.register(EVENT_TIMER, self.timed_reboot)
            self.reboot_timer_started = True
    # ----------------------------------------------------------------------------------------------------
    def timed_reboot(self, event: Event):
        """
        删除过期合约后一分钟重启交易子进程
        """
        if not self.reboot_datetime:
            return
        if datetime.now(TZ_INFO) - self.reboot_datetime < timedelta(seconds=60):
            return
        write_log(f"交易接口：{self.reboot_gateway_name}删除过期合约后即将重启")
        save_connection_status(self.reboot_gateway_name, False)
        EVENT_ENGINE.unregister(EVENT_TIMER, self.timed_reboot)
        self.reboot_timer_started = False
        self.reboot_datetime = None
    # ----------------------------------------------------------------------------------------------------
    def account_path(self, file_name:str) -> Path:
        """
        账户资金记录(csv)路径
        """
        account_path = PARENT2_PATH / "app" / "portfolio_strategy" / "account_info"
        if not account_path.exists():
            account_path.mkdir(parents=True, exist_ok=True)
        account_path = account_path / f"{file_name}.csv"
        return account_path 
    # ----------------------------------------------------------------------------------------------------
    def trade_path(self,app_name:str) -> Path:
        """
        合约交易记录(csv)路径
        """
        trade_path = PARENT2_PATH / "app" / app_name / "trade_info"
        if not trade_path.exists():
            trade_path.mkdir(parents=True, exist_ok=True)
        trade_path = trade_path / f"{self.now_date}_trade_info.csv"
        return trade_path
    # ----------------------------------------------------------------------------------------------------
    def cost_path(self,app_name:str) -> Path:
        """
        交易成本记录(csv)路径
        """
        cost_path = PARENT2_PATH / "app" / app_name / "cost_info"
        if not cost_path.exists():
            cost_path.mkdir(parents=True, exist_ok=True)
        cost_path = cost_path / f"{self.now_date}_cost_info.csv"
        return cost_path
    # ----------------------------------------------------------------------------------------------------
    def opt_path(self, strategy_name: str,app_name:str) -> Path:
        """
        多进程优化和遗传算法优化结果(html)保存路径
        """
        opt_path = PARENT2_PATH / "app" / app_name / "optimization_result"
        if not opt_path.exists():
            opt_path.mkdir(parents=True, exist_ok=True)
        opt_path = opt_path / f"{self.now_datetime}_optimization_{strategy_name}.html"
        return opt_path
    # ----------------------------------------------------------------------------------------------------
    def backtesting_path(self, strategy_name: str,app_name:str) -> Path:
        """
        回测结果(csv)保存路径
        """
        result_path = PARENT2_PATH / "app" / app_name / "backtesting_result"
        if not result_path.exists():
            result_path.mkdir(parents=True, exist_ok=True)
        result_path = result_path / f"{self.now_datetime}_backtesting_{strategy_name}.csv"
        return result_path
# ----------------------------------------------------------------------------------------------------
def get_price_ticks() -> Dict[str, float]:
    """
    获取合约 vt_symbol 和 price_tick 映射字典。

    该函数主要用于加载合约的最小价格变动单位（price_tick），并将其与合约标识符（vt_symbol）关联起来。
    如果缓存中没有该数据，则从合约数据中计算并保存。

    返回:
        price_ticks: Dict[str, float], 包含合约 vt_symbol 和相应 price_tick 的字典
    """
    file_name = "price_ticks"
    price_ticks = load_h5(file_name)
    if not price_ticks:
        data: Dict[str, ContractData] = load_h5("contract_data")
        price_ticks = {vt_symbol: contract.price_tick for vt_symbol, contract in data.items()}
        save_h5(file_name, price_ticks)
    return price_ticks
# ----------------------------------------------------------------------------------------------------
#数字货币交易接口
currency_gateway_names = [
    "OKX", "HUOBIF", "HUOBIS", "HUOBISUSDT", "BINANCEF", "BINANCES", "GATEIOS", "GATEIOUSDT", "KUCOIN", "DYDX", "BYBITONE", "BITGETONE","BINGXS",
    "DERIBIT","ORANGEX","HYPERLIQUID","POLYMARKET"
]
# ----------------------------------------------------------------------------------------------------
class SendFile:
    """
    * 钉钉发送文件
    * 需要钉钉后台绑定ip地址https://open-dev.dingtalk.com/fe/app#/appMgr/inner/eapp/1484669569/2
    * https://open-dev.dingtalk.com/，获取CorpId(网页右上角)
    * 进入开发者后台应用开发/企业内部应用/钉钉应用，选中应用点击右侧扩展菜单的应用详情，点击左上角的凭证与基础信息获取Client ID，Client Secret
    * 进入https://open.dingtalk.com/tools/explorer/jsapi?id=10303，左侧栏选择chooseChat，手机钉钉扫右侧二维码，登录钉钉开发者后台之后corpId默认填充不用填，isAllowCreateGroup和filterNotOwnerGroup选False，右侧点运行调试，在手机上选中接受文件的群获取chatid
    """
    # ----------------------------------------------------------------------------------------------------
    def __init__(self):
        self.client_id = "xxx"  # 用户id
        self.client_secret = "xxx"  # 用户密码
        self.chat_id = "xxx"
    # ----------------------------------------------------------------------------------------------------
    def get_access_token(self):
        """
        获取接口凭证
        """
        url = f"https://oapi.dingtalk.com/gettoken?appkey={self.client_id}&appsecret={self.client_secret}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("access_token")
        except requests.RequestException as err:
            write_log(f"获取access_token失败，错误信息：{err}")
            return None
    # ----------------------------------------------------------------------------------------------------
    def get_media_id(self, file_path: str):
        """
        获取文件media_id
        """
        self.access_token = self.get_access_token()    # 接口凭证，值不固定必须每次使用前获取
        url = f"https://oapi.dingtalk.com/media/upload?access_token={self.access_token}&type=file"
        try:
            with open(file_path, "rb") as file:
                files = {"media": file}
                response = requests.post(url, files=files)
                response.raise_for_status()
                data = response.json()
                if data["errcode"]:
                    write_log(f"获取media_id失败，错误信息：{data['errmsg']}")
                    return ""
                return data["media_id"]
        except requests.RequestException as err:
            write_log(f"上传文件失败，错误信息：{err}")
            return ""
    # ----------------------------------------------------------------------------------------------------
    def send_file(self, file_path: str):
        """
        * 发送文件到钉钉
        """
        media_id = self.get_media_id(file_path)
        if not media_id:
            return

        url = f"https://oapi.dingtalk.com/chat/send?access_token={self.access_token}"
        payload = {"chatid": self.chat_id, "msg": {"msgtype": "file", "file": {"media_id": media_id}}}

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            response.raise_for_status()
            data = response.json()
            if data["errcode"]:
                write_log(f"发送文件出错，错误代码：{data['errcode']}，错误信息：{data['errmsg']}")
        except requests.RequestException as err:
            write_log(f"发送文件请求出错，错误信息：{err}")

