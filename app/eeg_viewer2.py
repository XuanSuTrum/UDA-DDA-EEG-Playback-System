import sys
import threading
import queue
import serial
import serial.tools.list_ports
import numpy as np
import re
import csv
import os
import socket
import time
import traceback
from datetime import datetime
from collections import deque
from scipy import signal as scipy_signal
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QComboBox, QPushButton, QLabel, QFrame,
                             QSplitter, QCheckBox, QLineEdit, QTextEdit, QScrollArea,
                             QGroupBox, QGridLayout, QSlider, QProgressBar, QFileDialog,
                             QSizePolicy, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QTextCursor, QColor, QBrush
import pyqtgraph as pg

# ============ 引入决策引擎和模型适配器 ============
try:
    from model_adapters import (
        AdapterFactory,
        EEGNetAdapter,
        GenericDeepLearningAdapter,
        SVMAdapter,
        UDADDAOnlineAdapter,
    )
    MODEL_ADAPTERS_IMPORT_ERROR = None
except Exception as exc:
    AdapterFactory = None
    SVMAdapter = None
    GenericDeepLearningAdapter = None
    EEGNetAdapter = None
    UDADDAOnlineAdapter = None
    MODEL_ADAPTERS_IMPORT_ERROR = exc

# 尝试导入SVM相关模块（如果失败会打印警告）
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False
    print("警告: 未安装 sklearn 或 joblib，SVM功能将不可用")

# 尝试导入scipy用于加载MAT文件
try:
    from scipy.io import loadmat
    MAT_AVAILABLE = True
except ImportError:
    MAT_AVAILABLE = False
    print("警告: 未安装 scipy，MAT文件导入功能将不可用")

# ============ 配置参数 ============
# UDP配置
UDP_IP = "127.0.0.1"  # 目标IP（下位机）
UDP_PORT = 8888       # 目标端口

# SVM配置
SVM_MODEL_PATH = "svm_de_8ch_baseline_model.pkl"  # SVM模型文件路径（SEED DE模型，包含scaler）
FEATURE_WINDOW_SIZE = 200                       # 特征提取窗口大小（200Hz采样率，1秒窗口）
SAMPLE_RATE = 200                               # 采样率 200 Hz
NEGATIVE_THRESHOLD = 0.5
NEGATIVE_SCORE_EMA_ALPHA = 0.10
DISPLAY_DECISION_WINDOW = 1
UI_DECISION_UPDATE_EVERY_N_WINDOWS = 1
INFERENCE_SUBMIT_INTERVAL_SEC = 5.0
PREDICTION_SYNC_SOURCE_TYPE = "prediction-sync-lds"

# 62通道离线回放时，模型推理使用完整62通道，界面仍只显示这8个通道。
DISPLAY_CHANNEL_NAMES = ["FP1", "FP2", "F3", "F4", "T7", "T8", "P3", "P4"]
DISPLAY_CHANNEL_INDICES_62 = [0, 2, 7, 11, 23, 31, 43, 47]

# 状态机配置
REFRACTORY_PERIOD_MS = 60000      # 不应期时长（毫秒）

# ============ 信号发射器，用于跨线程通信 ============
class SerialSignals(QObject):
    data_received = pyqtSignal(list)
    connection_status = pyqtSignal(bool, str)

class LogSignals(QObject):
    new_log = pyqtSignal(str, str)  # (message, color)

class FSMSignals(QObject):
    state_changed = pyqtSignal(str)  # 新状态名称
    stimulus_triggered = pyqtSignal(str)  # 触发刺激信息

class DecisionSignals(QObject):
    """决策引擎信号（传递完整结果字典）"""
    decision_result = pyqtSignal(dict)  # 完整的决策结果字典

# ============ 日志记录器（线程安全） ============
class LogEmitter(QObject):
    """日志发射器，确保多线程安全"""
    def __init__(self):
        super().__init__()
        self.log_signals = LogSignals()

    def log(self, message, level="info"):
        """发送日志消息（可从任意线程调用）"""
        # 根据级别设置颜色
        colors = {
            "info": "#3498db",      # 蓝色
            "success": "#27ae60",   # 绿色
            "warning": "#f39c12",   # 橙色
            "error": "#e74c3c",     # 红色
            "fsm": "#9b59b6",       # 紫色（状态机）
            "stimulus": "#e74c3c",  # 红色（刺激）
            "calib": "#f59e0b",     # 橙色（标定）
        }
        color = colors.get(level, "#ecf0f1")
        self.log_signals.new_log.emit(message, color)

# 全局日志发射器
log_emitter = LogEmitter()

# ============ 微分熵计算函数 ============
def calculate_differential_entropy(data, sample_rate=250):
    """
    计算滑窗微分熵特征（基于频带功率）

    参数:
        data: 1D数组，信号数据
        sample_rate: 采样率(Hz)，默认250Hz

    返回:
        dict: 各频带的微分熵 {'delta': h1, 'theta': h2, 'alpha': h3, 'beta': h4}
    """
    from scipy import signal
    from scipy.fft import fft, fftfreq

    # 计算功率谱密度
    n = len(data)
    fft_vals = fft(data)
    fft_freq = fftfreq(n, 1.0 / sample_rate)

    # 只取正频率部分
    positive_freq_idx = fft_freq > 0
    fft_freq = fft_freq[positive_freq_idx]
    fft_vals = fft_vals[positive_freq_idx]

    # 功率谱
    power = np.abs(fft_vals) ** 2

    # 定义频带
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }

    entropy_values = {}

    for band_name, (low, high) in bands.items():
        # 提取频带内的功率
        band_mask = (fft_freq >= low) & (fft_freq <= high)
        band_power = power[band_mask]

        # 计算频带总功率
        total_power = np.sum(band_power)

        if total_power > 0:
            # 归一化功率谱密度
            psd = band_power / total_power

            # 计算微分熵: H = -sum(p * log(p))
            # 添加小值避免log(0)
            psd = psd + 1e-10
            entropy = -np.sum(psd * np.log(psd))
        else:
            entropy = 0.0

        entropy_values[band_name] = entropy

    return entropy_values

# ============ 决策引擎推理线程 ============
class ModelInferenceThread(threading.Thread):
    """
    模型推理线程（简化版，无FSM状态机）

    功能：
    1. 使用当前选中的适配器进行推理
    2. 直接传递预测结果，不做自动触发判断
    3. 通过信号通知主线程更新UI
    """
    def __init__(self, model_type: str = 'svm'):
        super().__init__()
        self.mutex = QMutex()
        self.daemon = True

        # 信号发射器
        self.signals = DecisionSignals()

        # 当前适配器（默认为 SVM）
        self.adapter = None
        self.model_type = model_type
        self.model_path = None  # 将在 _init_adapter 中使用默认路径

        # 推理队列
        self.inference_queue = queue.Queue(maxsize=1)
        self.running = False
        self.is_busy = False
        self.inference_submit_count = 0
        self.inference_run_count = 0
        self.inference_emit_count = 0

        # 初始化默认适配器
        self._init_adapter()

    def _init_adapter(self):
        """Initialize the selected model adapter."""
        if self.model_type == PREDICTION_SYNC_SOURCE_TYPE:
            self.adapter = None
            log_emitter.log("预测来源为后台 prediction CSV，同步展示模式不加载实时模型适配器。", "info")
            return
        if AdapterFactory is None:
            self.adapter = None
            log_emitter.log(
                f"实时模型适配器不可用，离线预测同步模式仍可使用: {MODEL_ADAPTERS_IMPORT_ERROR}",
                "warning",
            )
            return
        try:
            self.adapter = AdapterFactory.create_adapter(
                model_type=self.model_type,
                model_path=self.model_path,
                log_callback=lambda msg: log_emitter.log(msg, "model")
            )
            actual_path = self.adapter.model_path if self.adapter else "unknown"
            log_emitter.log(f"Loaded model: {self.model_type} ({actual_path})", "info")
            if self.adapter is not None:
                engine = getattr(self.adapter, "engine", None)
                log_emitter.log(f"Adapter created: {self.adapter.__class__.__name__}", "model")
                log_emitter.log(f"model_type: {self.model_type}", "model")
                log_emitter.log(f"model_path: {actual_path}", "model")
                if engine is not None:
                    log_emitter.log(f"using_identity_scaler: {getattr(engine, 'using_identity_scaler', None)}", "model")
        except Exception as e:
            log_emitter.log(f"Failed to load model: {e}", "error")
            log_emitter.log(traceback.format_exc(), "error")
            self.adapter = None

    def set_model(self, model_type: str, model_path: str = None):
        """
        动态切换模型适配器

        Args:
            model_type: 模型类型 ('svm', 'eegnet', 'uda-dda-online', 'uda-dda-binary')
            model_path: 模型文件路径（可选，默认使用预设路径）
        """
        if model_type == 'uda-dda':
            model_type = 'uda-dda-online'
        if model_type == PREDICTION_SYNC_SOURCE_TYPE:
            self.model_type = model_type
            self.model_path = None
            self.adapter = None
            log_emitter.log("已切换到后台预测同步模式，不加载实时模型。", "success")
            return

        # 停止当前推理
        was_running = self.running
        if was_running:
            self.stop()

        # 更新模型配置
        self.model_type = model_type
        if model_path is not None:
            self.model_path = model_path
        else:
            # 使用默认路径
            model_paths = {
                'svm': SVM_MODEL_PATH,
                'eegnet': 'eegnet_model.pth',
                'uda-dda': None,
                'sdc-net': 'sdc_net_model.pth',
                'uda-dda-online': None,
                'uda-dda-binary': None
            }
            self.model_path = model_paths.get(model_type, SVM_MODEL_PATH)

        # 重新初始化适配器
        self._init_adapter()

        # 恢复推理
        if was_running:
            self.running = True
            if not self.is_alive():
                self.start()

    def do_inference(self, eeg_matrix):
        """Submit the latest EEG window for asynchronous model inference."""
        if self.is_busy:
            if self.inference_submit_count <= 5 or self.inference_submit_count % 20 == 0:
                log_emitter.log("模型正在推理，跳过本次提交。", "model")
            return

        self.inference_submit_count += 1
        if self.inference_submit_count <= 5 or self.inference_submit_count % 20 == 0:
            log_emitter.log("已提交推理窗口。", "model")

        try:
            self.inference_queue.put_nowait(eeg_matrix)
        except queue.Full:
            try:
                while True:
                    self.inference_queue.get_nowait()
            except queue.Empty:
                pass
            self.inference_queue.put_nowait(eeg_matrix)
            log_emitter.log("推理队列已满，已丢弃旧窗口并保留最新窗口。", "warning")

    def run(self):
        """Model inference main loop."""
        self.running = True

        while self.running:
            try:
                eeg_matrix = self.inference_queue.get(timeout=0.1)
                self.inference_run_count += 1
                should_log = self.inference_run_count <= 5 or self.inference_run_count % 20 == 0
                start_time = time.time()
                self.is_busy = True
                if should_log:
                    log_emitter.log(f"开始模型推理，输入 shape: {getattr(eeg_matrix, 'shape', None)}", "model")
                    log_emitter.log(f"adapter class: {self.adapter.__class__.__name__ if self.adapter else None}", "model")

                if self.adapter is not None:
                    score, ui_emotion, class_idx, probs = self.adapter.predict(eeg_matrix, fs=SAMPLE_RATE)
                else:
                    probs = np.random.dirichlet([1, 1, 1])
                    score = probs[0] * 100
                    class_idx = int(np.argmax(probs))
                    emotion_map = ["Negative", "Neutral", "Positive"]
                    ui_emotion = emotion_map[class_idx]

                elapsed = time.time() - start_time
                probs_array = np.asarray(probs, dtype=float)
                log_emitter.log(
                    f"模型推理完成，耗时: {elapsed:.2f} 秒 | score={float(score):.4f} | "
                    f"ui_emotion={ui_emotion} | probs={probs_array.tolist()}",
                    "model"
                )

                result = {
                    "score": score,
                    "ui_emotion": ui_emotion,
                    "class_idx": class_idx,
                    "probabilities": probs_array,
                    "trigger": False,
                    "state": "DIRECT",
                    "timestamp": time.time()
                }

                self.signals.decision_result.emit(result)
                self.inference_emit_count += 1
                log_emitter.log("已发送 decision_result", "model")

            except queue.Empty:
                continue
            except Exception as e:
                log_emitter.log(f"[ModelInferenceThread] 推理异常: {e}", "error")
                log_emitter.log(traceback.format_exc(), "error")
            finally:
                self.is_busy = False

    def stop(self):
        """停止推理线程"""
        self.running = False

# ============ UDP通信模块 ============
class UDPSender:
    """UDP发送器（线程安全）"""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.mutex = QMutex()

    def init(self):
        """初始化UDP socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            log_emitter.log(f"UDP通信初始化成功: {self.ip}:{self.port}", "success")
            return True
        except Exception as e:
            log_emitter.log(f"UDP初始化失败: {str(e)}", "error")
            return False

    def send_stimulus(self, command="STIMULATE"):
        """发送刺激命令"""
        QMutexLocker(self.mutex)
        if self.socket is None:
            log_emitter.log("UDP未初始化，无法发送命令", "error")
            return False

        try:
            message = command.encode('utf-8')
            self.socket.sendto(message, (self.ip, self.port))
            log_emitter.log(f"UDP刺激命令已发送: {command} -> {self.ip}:{self.port}", "stimulus")
            return True
        except Exception as e:
            log_emitter.log(f"UDP发送失败: {str(e)}", "error")
            return False

    def close(self):
        """关闭socket"""
        if self.socket:
            self.socket.close()

# ============ 串口读取线程 ============
class SerialReaderThread(threading.Thread):
    def __init__(self, port, baudrate=9600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn = None
        self.signals = SerialSignals()
        self.daemon = True
        self.buffer = ""  # 数据缓冲区
        self.data_batch = []  # 批量发送队列
        self.batch_size = 10  # 每10个数据包批量发送

    def run(self):
        self.running = True
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.signals.connection_status.emit(True, f"已连接到 {self.port}")

            while self.running:
                try:
                    # 读取所有可用数据
                    if self.serial_conn.in_waiting > 0:
                        raw_data = self.serial_conn.read(self.serial_conn.in_waiting)
                        self.buffer += raw_data.decode('utf-8', errors='ignore')

                        # 处理缓冲区中的所有完整数据包
                        while True:
                            # 查找$作为开始
                            start_idx = self.buffer.find('$')
                            if start_idx == -1:
                                # 没有开始标记，清空缓冲区
                                self.buffer = ""
                                break

                            # 查找对应的;作为结束标记
                            end_idx = self.buffer.find(';', start_idx)
                            if end_idx != -1:
                                # 提取完整的数据包（不含$和;）
                                data_str = self.buffer[start_idx + 1:end_idx]

                                # 移除已处理的数据
                                self.buffer = self.buffer[end_idx + 1:]

                                # 用正则表达式提取所有数字
                                vals = re.findall(r"[-+]?\d*\.?\d+", data_str)

                                if len(vals) >= 1:  # 至少有1个通道
                                    try:
                                        values = [float(v) for v in vals]
                                        # 添加到批量队列
                                        self.data_batch.append(values)

                                        # 达到批量大小，发送所有数据
                                        if len(self.data_batch) >= self.batch_size:
                                            for data in self.data_batch:
                                                self.signals.data_received.emit(data)
                                            self.data_batch.clear()
                                    except ValueError:
                                        pass
                            else:
                                # 没有找到结束标记，数据包不完整，等待更多数据
                                break

                        # 如果缓冲区太大，清理一下（防止内存溢出）
                        if len(self.buffer) > 10000:
                            # 保留最后一个可能不完整的数据包
                            last_dollar = self.buffer.rfind('$')
                            if last_dollar != -1:
                                self.buffer = self.buffer[last_dollar:]
                            else:
                                self.buffer = ""
                    else:
                        # 没有新数据时，发送剩余的批量数据
                        if self.data_batch:
                            for data in self.data_batch:
                                self.signals.data_received.emit(data)
                            self.data_batch.clear()
                        # 短暂休眠，减少CPU占用
                        time.sleep(0.001)

                except Exception as e:
                    if self.running:
                        print(f"读取错误: {e}")
                    break
        except Exception as e:
            self.signals.connection_status.emit(False, f"连接失败: {str(e)}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.signals.connection_status.emit(False, "串口已关闭")

    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

# ============ 离线回放线程 ============
class OfflinePlaybackThread(threading.Thread):
    """离线数据回放线程（支持MAT和CSV格式）"""
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        self.signals = SerialSignals()  # 复用相同的信号
        self.daemon = True

        # 数据
        self.data = []  # 数据列表（每行是一个通道值列表）
        self.current_index = 0

        # 播放控制
        self.playback_speed = 1.0  # 播放速度倍率
        self.base_interval = 0.005  # 基础间隔（秒），对应200Hz采样率

    def load_data(self, filepath):
        """从文件加载数据（支持MAT和CSV格式）"""
        try:
            # 判断文件类型
            if filepath.endswith('.mat'):
                return self._load_mat(filepath)
            elif filepath.endswith('.csv'):
                return self._load_csv(filepath)
            else:
                # 尝试按MAT加载
                return self._load_mat(filepath)
        except Exception as e:
            log_emitter.log(f"加载数据失败: {str(e)}", "error")
            return False, 0, 0

    def _load_mat(self, filepath):
        """从MAT文件加载数据"""
        if not MAT_AVAILABLE:
            log_emitter.log("MAT功能不可用: 缺少 scipy 依赖", "error")
            return False, 0, 0

        try:
            mat_data = loadmat(filepath)

            # 查找数据变量。上位机62通道回放文件优先使用 replay_data。
            data_key = None
            if "replay_data" in mat_data:
                data_key = "replay_data"
            else:
                for key in mat_data.keys():
                    if key.startswith('__'):
                        continue
                    value = mat_data[key]
                    if hasattr(value, "ndim") and value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                        data_key = key
                        break

            if data_key is None:
                log_emitter.log(f"MAT文件未找到数据: {filepath}", "error")
                return False, 0, 0

            raw_data = mat_data[data_key]

            # 检查数据格式: (通道, 数据点) 或 (数据点, 通道)
            if raw_data.ndim == 2:
                # 判断哪一维是通道
                # 通常通道数较少(<=32)，数据点较多
                if raw_data.shape[0] <= raw_data.shape[1]:
                    # 格式: (通道, 数据点)
                    num_channels = raw_data.shape[0]
                    num_points = raw_data.shape[1]
                    # 转置为 (数据点, 通道) 便于处理
                    raw_data = raw_data.T
                else:
                    # 格式: (数据点, 通道)
                    num_channels = raw_data.shape[1]
                    num_points = raw_data.shape[0]
            else:
                log_emitter.log(f"MAT数据格式不支持: 维度={raw_data.ndim}", "error")
                return False, 0, 0

            # 转换为列表格式
            self.data = []
            for i in range(num_points):
                self.data.append([float(raw_data[i, ch]) for ch in range(num_channels)])

            self.current_index = 0

            log_emitter.log(f"成功加载MAT: {os.path.basename(filepath)}", "success")
            log_emitter.log(f"  变量名: {data_key}, 数据点: {num_points}, 通道数: {num_channels}", "info")

            return True, num_channels, num_points

        except Exception as e:
            log_emitter.log(f"加载MAT失败: {str(e)}", "error")
            return False, 0, 0

    def _load_csv(self, filepath):
        """从CSV文件加载数据"""
        try:
            data = []
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)

                def parse_numeric_row(row):
                    if not row:  # 跳过空行
                        return None
                    values = [float(v) for v in row]
                    # 兼容旧记录CSV: Time + 通道数据。
                    if len(values) in (9, 63):
                        return values[1:]
                    return values

                if first_row is not None:
                    try:
                        values = parse_numeric_row(first_row)
                        if values:
                            data.append(values)
                    except ValueError:
                        # 第一行是表头，跳过。
                        pass

                for row in reader:
                    try:
                        values = parse_numeric_row(row)
                        if values:
                            data.append(values)
                    except ValueError:
                        continue

            self.data = data
            self.current_index = 0

            if data:
                num_channels = len(data[0])
                num_rows = len(data)
                log_emitter.log(f"成功加载CSV: {os.path.basename(filepath)}", "success")
                log_emitter.log(f"  数据行数: {num_rows}, 通道数: {num_channels}", "info")
                return True, num_channels, num_rows
            else:
                log_emitter.log(f"CSV文件为空: {filepath}", "error")
                return False, 0, 0

        except Exception as e:
            log_emitter.log(f"加载CSV失败: {str(e)}", "error")
            return False, 0, 0

    def set_playback_speed(self, speed):
        """设置播放速度"""
        self.playback_speed = speed

    def set_position(self, index):
        """设置播放位置"""
        self.current_index = max(0, min(index, len(self.data) - 1))

    def pause(self):
        """暂停播放"""
        self.paused = True

    def resume(self):
        """继续播放"""
        self.paused = False

    def run(self):
        """主播放循环"""
        self.running = True
        self.paused = False
        self.signals.connection_status.emit(True, f"离线回放模式: {len(self.data)} 点")

        while self.running and self.current_index < len(self.data):
            if not self.paused:
                # 发送当前数据
                values = self.data[self.current_index]
                self.signals.data_received.emit(values)

                # 更新进度信号（复用connection_status传递进度）
                progress = int((self.current_index + 1) / len(self.data) * 100)
                self.signals.connection_status.emit(True, f"进度:{progress}%")

                self.current_index += 1

                # 计算睡眠时间（根据播放速度）
                sleep_time = self.base_interval / self.playback_speed
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)  # 暂停时短暂休眠

        # 播放结束
        if self.current_index >= len(self.data):
            self.signals.connection_status.emit(False, "回放结束")
            log_emitter.log("离线回放完成", "success")

    def stop(self):
        """停止播放"""
        self.running = False
        self.paused = False

# ============ EEG数据显示主窗口 ============
class EEGDisplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ============ 串口相关 ============
        self.serial_thread = None
        self.stim_serial = None  # 刺激器串口对象
        self.playback_thread = None  # 离线回放线程
        self.is_paused = False
        self.num_channels = 0  # 通道数量，初始未知
        self.input_num_channels = 0  # 原始输入通道数，可为8或62
        self.display_num_channels = 0  # 界面显示通道数，62通道回放时仍为8
        self.data_buffers = []  # 数据缓冲区，动态创建
        self.full_data_buffers = []  # 完整输入通道缓冲区，供模型推理
        self.plots = []  # 绘图部件列表
        self.curves = []  # 曲线列表
        self.ptr = 0
        self.plot_layout = None  # 绘图布局

        # ============ 离线回放相关 ============
        self.is_playback_mode = False  # 是否为回放模式
        self.playback_file_path = ""  # 回放文件路径
        self.playback_total_rows = 0  # 回放数据总行数
        self.prediction_df = []
        self.prediction_times = np.array([], dtype=float)
        self.prediction_mode_enabled = False
        self.current_prediction_index = 0
        self.prediction_csv_path = ""
        self.last_prediction_sync_log_time = 0.0

        # ============ 微分熵相关 ============
        self.entropy_window_size = 200  # 滑窗大小（采样点数，1秒@200Hz）
        self.entropy_values = {}  # 存储各通道的熵值
        self.entropy_update_counter = 0  # 熵值更新计数器

        # ============ 0Ω修正开关状态 ============
        self.zero_ohm_correction = False  # 默认关闭（正常测真人脑电模式）

        # ============ Y轴量程设置 ============
        self.y_axis_range = 'auto'  # 默认自动缩放
        self.y_range_limits = {
            '50': 50,
            '100': 100,
            '250': 250,
            '500': 500,
            'auto': None  # 自动缩放
        }

        # ============ 数据保存相关 ============
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.recording_start_time = None

        # ============ MAT 会话保存模块 ============
        self.session_manager = self._create_session_manager()
        self.mat_session_active = False  # MAT session是否活动

        # ============ 物理通道映射相关 ============
        self.physical_channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']

        # 统一的通道映射（默认配置，用户可修改）
        self.channel_mapping = {
            'CH1': 'Fp1', 'CH2': 'Fp2', 'CH3': 'F3', 'CH4': 'F4',
            'CH5': 'T7', 'CH6': 'T8', 'CH7': 'P3', 'CH8': 'P4'
        }

        # 映射配置UI组件
        self.channel_mapping_combos = []  # 存储8个下拉框
        self.confirm_mapping_btn = None   # 确认映射按钮

        # ============ 离线回放评测相关 ============
        self.current_true_label = None  # 当前文件的真实标签（0:负向, 1:中性, 2:正向）
        self.online_predictions = []  # 在线预测结果列表

        # ============ UI 优化相关 ============
        # 系统日志降频与降噪
        self.inference_counter = 0  # 推理计数器
        self.log_suppression_factor = 4  # 日志降频因子（每4次推理打印一次，即每2秒）
        self.recent_scores = []  # 最近几次推理的得分（用于计算平均）
        self.recent_emotions = []  # 最近几次推理的情绪（用于统计多数派）
        self.negative_threshold = NEGATIVE_THRESHOLD
        self.display_decision_window = DISPLAY_DECISION_WINDOW
        self.negative_score_ema_alpha = NEGATIVE_SCORE_EMA_ALPHA
        self.smoothed_negative_score = None  # 仅保留兼容/debug，不再作为主显示依据
        self.decision_window_counter = 0
        self.ui_decision_update_every_n_windows = UI_DECISION_UPDATE_EVERY_N_WINDOWS
        self.recent_negative_probs = deque(maxlen=self.display_decision_window)
        # 频带能量平滑化（滑动平均队列）
        self.band_energy_queue = deque(maxlen=10)  # 长度为10的队列（平滑过去2秒的能量）
        self.band_energy_queue_initialized = False  # 队列是否已初始化

        # ============ 其他状态 ============
        self.packet_count = 0  # 数据包计数器
        self.pause_btn = None  # 暂停按钮（延迟在init_ui中创建）
        self.stimulus_count = 0  # 刺激次数
        self.start_time = None  # 启动时间
        self.threshold_index = 0  # 阈值图索引

        # ============ EEG 实时滤波流水线 ============
        self.fs = 200.0  # 采样率 (Hz) - 与离线训练对齐

        # 50Hz 陷波器参数（滤除市电干扰）
        self.f0 = 50.0  # 陷波频率
        self.Q = 30.0   # 品质因数
        self.b_notch, self.a_notch = iirnotch(self.f0 / (self.fs / 2), self.Q)

        # 0.5-50Hz 带通滤波器参数（滤除基线漂移和高频噪声）
        self.lowcut = 0.5  # 低频截止频率
        self.highcut = 50.0  # 高频截止频率
        self.order = 4  # 滤波器阶数
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b_band, self.a_band = butter(self.order, [low, high], btype='band')

        # 滤波后的数据缓冲区（用于显示和特征提取）
        self.filtered_buffers = []  # 与 data_buffers 对应的滤波后数据
        self.full_filtered_buffers = []  # 与 full_data_buffers 对应，供模型推理
        self.last_inference_shape_log = 0
        self.inference_submit_interval_sec = INFERENCE_SUBMIT_INTERVAL_SEC
        self.last_inference_submit_time = 0.0
        self.last_inference_skip_log_time = 0.0

        # ============ 信号发射器 ============
        self.fsm_signals = FSMSignals()

        # ============ 初始化UI ============
        self.init_ui()
        self.refresh_ports()

        # ============ 决策引擎和UDP ============
        # 从 UI 选择器获取初始模型类型（默认为 SVM）
        initial_model_type = self.model_selector_combo.currentData()
        self.model_thread = ModelInferenceThread(model_type=initial_model_type)
        self.model_thread.start()  # 启动推理线程
        self.udp_sender = UDPSender(UDP_IP, UDP_PORT)
        self.udp_sender.init()

        # ============ 连接信号 ============
        log_emitter.log_signals.new_log.connect(self.append_log)
        self.fsm_signals.stimulus_triggered.connect(self.on_stimulus_triggered)

        # 连接决策引擎结果信号到处理函数
        self.model_thread.signals.decision_result.connect(self.handle_decision_result)

        # ============ 启动日志 ============
        log_emitter.log("========================================", "info")
        log_emitter.log("脑电数据实时显示系统 v2.0", "info")
        log_emitter.log("CPS闭环调控边缘计算上位机", "info")
        log_emitter.log("========================================", "info")
        log_emitter.log(f"决策引擎初始化: 进入标定状态 (CALIBRATION)", "calib")

    def _create_session_manager(self):
        """Create MAT session manager, or a no-op fallback when core package is absent."""
        try:
            from core.session_manager import get_session_manager
            return get_session_manager()
        except ModuleNotFoundError as exc:
            if exc.name != "core":
                raise

            class NullSessionManager:
                session_id = "session_manager_unavailable"

                def start_session(self, *args, **kwargs):
                    return None

                def end_session(self, *args, **kwargs):
                    return None

                def update_channel_info(self, *args, **kwargs):
                    return None

                def append_event(self, *args, **kwargs):
                    return None

                def append_prediction(self, *args, **kwargs):
                    return None

                def append_stim_record(self, *args, **kwargs):
                    return None

                def append_eeg(self, *args, **kwargs):
                    return None

            print("[WARN] core.session_manager not found; MAT session recording is disabled.")
            return NullSessionManager()

    def init_ui(self):
        self.setWindowTitle('NEURO-LINK // BCI 闭环调控系统')
        self.setGeometry(100, 50, 1920, 1080)  # 全屏尺寸

        # ============ 全局 pyqtgraph 配置（高级深色主题）============
        # 设置深空灰背景（替代纯黑）
        pg.setConfigOption('background', '#181a1f')
        # 设置前景色为亮灰色
        pg.setConfigOption('foreground', '#d7dae0')
        # 开启全局抗锯齿，让折线更平滑丝滑
        pg.setConfigOptions(antialias=True)

        # ============ 注入全局 QSS 皮肤（Premium Dark Theme）============
        self.setStyleSheet("""
            /* ============ 主窗口背景 ============*/
            QMainWindow {
                background-color: #21252b;
            }
            QWidget {
                background-color: transparent;
                color: #abb2bf;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            }

            /* ============ QGroupBox 面板外框 ============*/
            QGroupBox {
                color: #abb2bf;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #3e4451;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #282c34;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #abb2bf;
            }

            /* ============ QPushButton 现代按钮 ============*/
            QPushButton {
                background-color: #3a3f4b;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4b5260;
            }
            QPushButton:pressed {
                background-color: #2c313a;
            }
            QPushButton:disabled {
                background-color: #2c313a;
                color: #5c6370;
                border: none;
            }

            /* ============ QLabel 文本标签 ============*/
            QLabel {
                color: #abb2bf;
                background-color: transparent;
            }

            /* ============ QComboBox 下拉框 ============*/
            QComboBox {
                background-color: #181a1f;
                color: #ffffff;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #4b5260;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #3a3f4b;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #abb2bf;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #282c34;
                color: #abb2bf;
                border: 1px solid #3e4451;
                selection-background-color: #3a3f4b;
                selection-color: #ffffff;
            }

            /* ============ QScrollBar 滚动条 ============*/
            QScrollBar:vertical {
                background-color: #181a1f;
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3f4b;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4b5260;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            /* ============ QSlider 滑块 ============*/
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a3f4b;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d4ff;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #33ddff;
            }

            /* ============ QTextEdit 文本框 ============*/
            QTextEdit {
                background-color: #181a1f;
                color: #d7dae0;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
            QTextEdit:focus {
                border: 1px solid #00d4ff;
            }

            /* ============ QLineEdit 输入框 ============*/
            QLineEdit {
                background-color: #181a1f;
                color: #d7dae0;
                border: 1px solid #3e4451;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
            }

            /* ============ QCheckBox 复选框 ============*/
            QCheckBox {
                color: #abb2bf;
                spacing: 8px;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3e4451;
                border-radius: 3px;
                background-color: #181a1f;
            }
            QCheckBox::indicator:checked {
                background-color: #00d4ff;
                border: 2px solid #00d4ff;
            }
            QCheckBox::indicator:checked::after {
                content: '✓';
                color: #ffffff;
            }
        """)

        # 主部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)  # 增加面板之间的间隔
        main_layout.setContentsMargins(15, 15, 15, 15)  # 增加外边距

        # ============ 顶部全局控制栏 ============
        top_bar = QWidget()
        top_bar.setFixedHeight(80)
        top_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a0e27, stop:0.5 #1a1a2e, stop:1 #0f0f23);
                border-bottom: 2px solid #00d4ff;
            }
        """)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setSpacing(20)
        top_bar_layout.setContentsMargins(20, 10, 20, 10)

        # Logo和标题
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setSpacing(2)
        title_label = QLabel('NEURO-LINK')
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
            font-family: 'Arial Black', sans-serif;
            letter-spacing: 2px;
        """)
        subtitle_label = QLabel('BCI 闭环调控系统 v2.0')
        subtitle_label.setStyleSheet("""
            font-size: 11px;
            color: #6c7a89;
            letter-spacing: 1px;
        """)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        top_bar_layout.addWidget(title_widget)

        # 分隔线
        separator = QLabel('|')
        separator.setStyleSheet('color: #2d3748; font-size: 20px;')
        top_bar_layout.addWidget(separator)

        # ============ 文件加载区 ============
        file_group = QWidget()
        file_group.setStyleSheet("background-color: rgba(0, 212, 255, 0.05); border-radius: 8px; padding: 8px;")
        file_layout = QHBoxLayout(file_group)
        file_layout.setSpacing(10)

        self.load_csv_btn = QPushButton('📂 加载数据')
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        self.load_csv_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d4ff, stop:1 #0088aa);
                color: #0a0e27;
            }
        """)
        file_layout.addWidget(self.load_csv_btn)

        self.file_info_label = QLabel('未加载数据')
        self.file_info_label.setStyleSheet('color: #718096; font-size: 12px;')
        file_layout.addWidget(self.file_info_label)

        top_bar_layout.addWidget(file_group)

        # ============ 连接控制区 ============
        conn_group = QWidget()
        conn_group.setStyleSheet("background-color: rgba(16, 185, 129, 0.05); border-radius: 8px; padding: 8px;")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.setSpacing(8)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(100)
        self.port_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #00d4ff;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #00d4ff;
            }
        """)
        conn_layout.addWidget(self.port_combo)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['9600', '19200', '38400', '57600', '115200'])
        self.baud_combo.setCurrentText('115200')
        self.baud_combo.setMinimumWidth(80)
        self.baud_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #00d4ff;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }
        """)
        conn_layout.addWidget(self.baud_combo)

        # ============ 预测模型选择区 ============
        model_group = QWidget()
        model_group.setStyleSheet("background-color: rgba(245, 158, 11, 0.05); border-radius: 8px; padding: 8px;")
        model_layout = QHBoxLayout(model_group)
        model_layout.setSpacing(8)

        model_label = QLabel('预测来源:')
        model_label.setStyleSheet('color: #f59e0b; font-size: 12px; font-weight: bold;')
        model_layout.addWidget(model_label)

        self.model_selector_combo = QComboBox()
        self.model_selector_combo.addItem('UDA-DDA 后台预测（DE+LDS, 62ch）', PREDICTION_SYNC_SOURCE_TYPE)
        self.model_selector_combo.setCurrentIndex(0)
        self.model_selector_combo.setMinimumWidth(220)
        self.model_selector_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #f59e0b;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #f59e0b;
            }
        """)
        # 连接信号：模型切换时触发
        self.model_selector_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_selector_combo)

        top_bar_layout.addWidget(model_group)

        self.refresh_btn = QPushButton('🔄')
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #00d4ff;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00d4ff;
                color: #0a0e27;
            }
        """)
        conn_layout.addWidget(self.refresh_btn)

        self.connect_btn = QPushButton('🔌 连接')
        self.connect_btn.clicked.connect(self.toggle_serial)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #10b981;
                border: 1px solid #10b981;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #10b981;
                color: #0a0e27;
            }
        """)
        conn_layout.addWidget(self.connect_btn)

        # 暂停按钮
        self.pause_btn = QPushButton('⏸')
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip('暂停/继续')
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setFixedWidth(40)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00d4ff;
                color: #0a0e27;
            }
            QPushButton:disabled {
                color: #4a5568;
                border: 1px solid #2d3748;
            }
        """)
        conn_layout.addWidget(self.pause_btn)

        top_bar_layout.addWidget(conn_group)

        # ============ 播放控制区 ============
        playback_group = QWidget()
        playback_group.setStyleSheet("background-color: rgba(139, 92, 246, 0.05); border-radius: 8px; padding: 8px;")
        playback_layout = QHBoxLayout(playback_group)
        playback_layout.setSpacing(8)

        self.play_pause_btn = QPushButton('▶')
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setFixedWidth(40)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #8b5cf6;
                border: 1px solid #8b5cf6;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8b5cf6;
                color: #fff;
            }
            QPushButton:disabled {
                color: #4a5568;
                border: 1px solid #2d3748;
            }
        """)
        playback_layout.addWidget(self.play_pause_btn)

        self.stop_playback_btn = QPushButton('⏹')
        self.stop_playback_btn.setEnabled(False)
        self.stop_playback_btn.clicked.connect(self.stop_playback)
        self.stop_playback_btn.setFixedWidth(40)
        self.stop_playback_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: #fff;
            }
            QPushButton:disabled {
                color: #4a5568;
                border: 1px solid #2d3748;
            }
        """)
        playback_layout.addWidget(self.stop_playback_btn)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.5x', '1x', '2x', '5x', '10x'])
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.setMinimumWidth(60)
        self.speed_combo.currentTextChanged.connect(self.on_playback_speed_changed)
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #8b5cf6;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        playback_layout.addWidget(self.speed_combo)

        # 进度条
        self.playback_progress = QProgressBar()
        self.playback_progress.setMinimumWidth(100)
        self.playback_progress.setMaximumWidth(150)
        self.playback_progress.setRange(0, 100)
        self.playback_progress.setValue(0)
        self.playback_progress.setTextVisible(True)
        self.playback_progress.setFormat('%v%')
        self.playback_progress.setStyleSheet("""
            QProgressBar {
                background-color: rgba(139, 92, 246, 0.1);
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #8b5cf6;
                font-size: 10px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 3px;
            }
        """)
        playback_layout.addWidget(self.playback_progress)

        # 进度滑块
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setMinimumWidth(80)
        self.playback_slider.setMaximumWidth(120)
        self.playback_slider.setRange(0, 100)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderPressed.connect(self.on_slider_pressed)
        self.playback_slider.sliderReleased.connect(self.on_slider_released)
        self.playback_slider.valueChanged.connect(self.on_slider_changed)
        self.playback_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(45, 55, 72, 0.5);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #8b5cf6;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)
        playback_layout.addWidget(self.playback_slider)

        top_bar_layout.addWidget(playback_group)

        # ============ 记录控制区 ============
        record_group = QWidget()
        record_group.setStyleSheet("background-color: rgba(239, 68, 68, 0.05); border-radius: 8px; padding: 8px;")
        record_layout = QHBoxLayout(record_group)
        record_layout.setSpacing(8)

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText('文件名')
        self.filename_edit.setMaximumWidth(120)
        self.filename_edit.setStyleSheet("""
            QLineEdit {
                background-color: #1a202c;
                color: #ecf0f1;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border: 1px solid #ef4444;
            }
        """)
        record_layout.addWidget(self.filename_edit)

        self.record_btn = QPushButton('● 记录')
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a202c;
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: #fff;
            }
        """)
        record_layout.addWidget(self.record_btn)

        top_bar_layout.addWidget(record_group)

        # ============ 状态指示器 ============
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setSpacing(15)

        # 连接状态
        conn_status_widget = self._create_status_indicator("连接", "#e74c3c")
        self.conn_status_label = conn_status_widget.findChild(QLabel, "status_text")
        self.conn_status_indicator = conn_status_widget.findChild(QLabel, "status_dot")
        status_layout.addWidget(conn_status_widget)

        # FSM状态
        fsm_status_widget = self._create_status_indicator("状态", "#f59e0b")
        self.fsm_status_label = fsm_status_widget.findChild(QLabel, "status_text")
        self.fsm_status_indicator = fsm_status_widget.findChild(QLabel, "status_dot")
        status_layout.addWidget(fsm_status_widget)

        top_bar_layout.addWidget(status_container)
        top_bar_layout.addStretch()

        main_layout.addWidget(top_bar)

        # ============ 主内容区布局（左右分割：主工作区 + 右侧栏）============
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)

        # ============ 左侧：主工作区（上方分析 + 下方时域波形）============
        main_workspace = QWidget()
        main_workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_workspace_layout = QVBoxLayout(main_workspace)
        main_workspace_layout.setSpacing(8)
        main_workspace_layout.setContentsMargins(5, 5, 5, 5)

        # ============ 主工作区上部：分析面板（左+中水平排列）============
        analysis_splitter = QSplitter(Qt.Horizontal)
        analysis_splitter.setHandleWidth(2)

        # ============ 左侧列：脑电电极分布 + 物理通道映射 ============
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(0)  # 取消组件间距
        left_layout.setContentsMargins(0, 0, 5, 0)  # 上下顶格，右边留5px间距

        # ============ 左侧：脑电电极分布与通道选择 ============
        eeg_widget = QWidget()
        eeg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 让它扩展填满可用空间
        eeg_widget.setMinimumHeight(450)  # 设置最小高度，防止被过度压缩
        eeg_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        eeg_layout = QVBoxLayout(eeg_widget)
        eeg_layout.setSpacing(10)
        eeg_layout.setContentsMargins(14, 6, 14, 6)  # 减小上下边距，让内容更顶格

        # 标题（高级感，科技蓝）
        eeg_title = QLabel('脑电电极分布 10-20 系统 // EEG ELECTRODE TOPOGRAPHY')
        eeg_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        eeg_layout.addWidget(eeg_title)

        # 电极分布图容器（卡片式设计，左右分割）
        eeg_map_container = QWidget()
        eeg_map_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        eeg_map_layout = QHBoxLayout(eeg_map_container)
        eeg_map_layout.setSpacing(12)
        eeg_map_layout.setContentsMargins(12, 12, 12, 12)

        # ============ 左侧：被试信息表单 ============
        subject_info_widget = QWidget()
        subject_info_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 6px;
                border: 1px solid #2a2e36;
            }
        """)
        subject_info_layout = QVBoxLayout(subject_info_widget)
        subject_info_layout.setSpacing(10)
        subject_info_layout.setContentsMargins(16, 12, 16, 12)

        # 标题
        subject_title = QLabel('被试信息 // SUBJECT INFO')
        subject_title.setStyleSheet("""
            font-size: 13px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        subject_info_layout.addWidget(subject_title)

        # 被试编号
        self.subject_id_input = QLineEdit()
        self.subject_id_input.setPlaceholderText('SUB01')
        self.subject_id_input.setStyleSheet("""
            QLineEdit {
                background-color: #252830;
                color: #cbd5e1;
                border: 1px solid #3f444e;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #60a5fa;
            }
        """)
        subject_info_layout.addWidget(QLabel('被试编号:'))
        subject_info_layout.addWidget(self.subject_id_input)

        # 年龄
        self.subject_age_input = QLineEdit()
        self.subject_age_input.setPlaceholderText('25')
        self.subject_age_input.setStyleSheet("""
            QLineEdit {
                background-color: #252830;
                color: #cbd5e1;
                border: 1px solid #3f444e;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #60a5fa;
            }
        """)
        subject_info_layout.addWidget(QLabel('年龄:'))
        subject_info_layout.addWidget(self.subject_age_input)

        # 性别
        self.subject_gender_combo = QComboBox()
        self.subject_gender_combo.addItems(['男', '女', '其他'])
        self.subject_gender_combo.setStyleSheet("""
            QComboBox {
                background-color: #252830;
                color: #cbd5e1;
                border: 1px solid #3f444e;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QComboBox:focus {
                border: 1px solid #60a5fa;
            }
        """)
        subject_info_layout.addWidget(QLabel('性别:'))
        subject_info_layout.addWidget(self.subject_gender_combo)

        # 日期
        from PyQt5.QtCore import QDate
        self.subject_date_input = QLineEdit()
        self.subject_date_input.setText(QDate.currentDate().toString('yyyy-MM-dd'))
        self.subject_date_input.setStyleSheet("""
            QLineEdit {
                background-color: #252830;
                color: #cbd5e1;
                border: 1px solid #3f444e;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #60a5fa;
            }
        """)
        subject_info_layout.addWidget(QLabel('日期:'))
        subject_info_layout.addWidget(self.subject_date_input)

        subject_info_layout.addStretch()
        eeg_map_layout.addWidget(subject_info_widget, 1)

        # ============ 右侧：10-20系统图 ============
        # 包装电极图
        eeg_plot_wrapper = QWidget()
        eeg_plot_wrapper.setStyleSheet("background-color: #14171a; border-radius: 6px;")
        eeg_plot_wrapper_layout = QVBoxLayout(eeg_plot_wrapper)
        eeg_plot_wrapper_layout.setContentsMargins(12, 12, 12, 12)

        # 右侧标题
        map_title = QLabel('10-20 系统图 // TOPOGRAPHY')
        map_title.setStyleSheet("""
            font-size: 13px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)
        eeg_plot_wrapper_layout.addWidget(map_title)

        # 左侧：电极分布图（头部俯视图）
        self.eeg_map_plot = pg.PlotWidget()
        self.eeg_map_plot.setBackground('#14171a')
        self.eeg_map_plot.showGrid(x=False, y=False)
        self.eeg_map_plot.setAspectLocked(True)
        self.eeg_map_plot.hideAxis('left')
        self.eeg_map_plot.hideAxis('bottom')
        self.eeg_map_plot.setMaximumHeight(320)
        self.eeg_map_plot.setMinimumHeight(300)
        self.eeg_map_plot.setContentsMargins(0, 0, 0, 0)

        # 定义32通道的电极位置（10-20系统，更精确的拓扑位置）
        self.eeg_positions = {
            'Fp1': (-0.5, 0.8), 'Fp2': (0.5, 0.8),
            'F7': (-0.7, 0.5), 'F3': (-0.4, 0.5), 'Fz': (0, 0.5), 'F4': (0.4, 0.5), 'F8': (0.7, 0.5),
            'FC5': (-0.5, 0.35), 'FC1': (-0.2, 0.35), 'FC2': (0.2, 0.35), 'FC6': (0.5, 0.35),
            'T7': (-0.8, 0.2), 'C3': (-0.4, 0.2), 'Cz': (0, 0.2), 'C4': (0.4, 0.2), 'T8': (0.8, 0.2),
            'CP5': (-0.5, 0.05), 'CP1': (-0.2, 0.05), 'CP2': (0.2, 0.05), 'CP6': (0.5, 0.05),
            'P7': (-0.7, -0.1), 'P3': (-0.4, -0.1), 'Pz': (0, -0.1), 'P4': (0.4, -0.1), 'P8': (0.7, -0.1),
            'PO3': (-0.2, -0.35), 'PO4': (0.2, -0.35),
            'O1': (-0.3, -0.6), 'Oz': (0, -0.6), 'O2': (0.3, -0.6)
        }

        # 存储电极点信息（添加 mapped 状态）
        self.eeg_electrode_items = {}  # {name: {'pos': (x,y), 'selected': bool, 'mapped': bool}}
        self.eeg_text_items = {}  # 存储文本项引用
        for name, (x, y) in self.eeg_positions.items():
            self.eeg_electrode_items[name] = {'pos': (x, y), 'selected': False, 'hovered': False, 'mapped': False}

        # 创建头部轮廓曲线
        self._draw_head_contour()

        # 绘制电极点（初始绘制）
        self.eeg_electrode_scatter = pg.ScatterPlotItem()
        self.eeg_map_plot.addItem(self.eeg_electrode_scatter)

        # 初始化文本标签
        for name, (x, y) in self.eeg_positions.items():
            text_item = pg.TextItem(text=name, color='#6b7280', anchor=(0.5, 1.5))
            text_item.setPos(x, y)
            text_item.setFont(QFont('Arial', 8))
            self.eeg_map_plot.addItem(text_item)
            self.eeg_text_items[name] = text_item

        # 初始绘制电极点
        self._update_eeg_map_display()

        eeg_plot_wrapper_layout.addWidget(self.eeg_map_plot)
        eeg_map_layout.addWidget(eeg_plot_wrapper, 2)

        eeg_layout.addWidget(eeg_map_container)

        # ============ 物理通道映射配置区域 ============
        self.channel_mapping_widget = QWidget()
        self.channel_mapping_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        mapping_layout = QVBoxLayout(self.channel_mapping_widget)
        mapping_layout.setSpacing(12)
        mapping_layout.setContentsMargins(16, 12, 16, 12)

        # 标题
        mapping_title = QLabel('物理通道映射配置 // CHANNEL MAPPING')
        mapping_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        mapping_layout.addWidget(mapping_title)

        # 说明文本
        mapping_hint = QLabel('为每个物理通道（CH1-8）选择对应的电极位置，不允许重复选择')
        mapping_hint.setStyleSheet("""
            font-size: 11px;
            color: #64748b;
            padding: 4px 0;
        """)
        mapping_layout.addWidget(mapping_hint)

        # 8个下拉框容器（网格布局，2列）
        mapping_grid = QGridLayout()
        mapping_grid.setSpacing(8)
        mapping_grid.setContentsMargins(0, 0, 0, 0)

        # 可选电极列表（包含"未使用"选项）
        electrode_options = ['未使用'] + list(self.eeg_positions.keys())

        # 创建8个下拉框
        self.channel_mapping_combos = []
        for i, ch_name in enumerate(self.physical_channels):
            # 行标签（物理通道名）
            ch_label = QLabel(ch_name)
            ch_label.setStyleSheet("""
                font-size: 12px;
                color: #cbd5e1;
                font-weight: 600;
                padding: 6px 8px;
                background-color: #252830;
                border-radius: 4px;
            """)

            # 下拉框
            combo = QComboBox()
            combo.addItems(electrode_options)
            combo.setCurrentText(self.channel_mapping[ch_name])
            combo.setStyleSheet("""
                QComboBox {
                    background-color: #252830;
                    color: #cbd5e1;
                    border: 1px solid #3f444e;
                    border-radius: 4px;
                    padding: 6px 10px;
                    font-size: 11px;
                }
                QComboBox:hover {
                    border: 1px solid #60a5fa;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox QAbstractItemView {
                    background-color: #1a1d23;
                    border: 1px solid #3f444e;
                    selection-background-color: #334155;
                    selection-color: #fff;
                }
            """)

            # 连接变化事件（防止重复选择）
            combo.currentTextChanged.connect(
                lambda text, idx=i, cb=combo: self.on_mapping_combo_changed(idx, cb, text)
            )

            self.channel_mapping_combos.append(combo)

            # 添加到网格（2列布局）
            row = i // 2
            col = (i % 2) * 2
            mapping_grid.addWidget(ch_label, row, col)
            mapping_grid.addWidget(combo, row, col + 1)

        mapping_layout.addLayout(mapping_grid)

        # 确认按钮
        self.confirm_mapping_btn = QPushButton('确认通道映射')
        self.confirm_mapping_btn.setEnabled(True)
        self.confirm_mapping_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
        """)
        self.confirm_mapping_btn.clicked.connect(self.confirm_channel_mapping)
        mapping_layout.addWidget(self.confirm_mapping_btn)

        # 添加到主布局
        eeg_layout.addWidget(self.channel_mapping_widget)

        # 鼠标移动事件处理（hover效果）
        self.eeg_map_plot.scene().sigMouseMoved.connect(self.on_eeg_map_hover)

        # 将eeg_widget添加到左侧面板（让它扩展填满整个可用高度）
        left_layout.addWidget(eeg_widget, 1)

        # 将left_panel添加到analysis_splitter
        analysis_splitter.addWidget(left_panel)

        # ============ 中间列：频带能量分析 + 情绪分类概率 ============
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(8)
        center_layout.setContentsMargins(5, 0, 0, 0)

        # ============ 中间：频带能量分析 ============
        band_widget = QWidget()
        band_widget.setMaximumHeight(380)  # 进一步扩展高度
        band_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        band_layout = QVBoxLayout(band_widget)
        band_layout.setSpacing(12)
        band_layout.setContentsMargins(16, 12, 16, 12)

        # 标题（统一风格）
        band_title = QLabel('频带能量分析 // POWER SPECTRAL DENSITY')
        band_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        band_layout.addWidget(band_title)

        # 图表容器（内层卡片）
        band_chart_container = QWidget()
        band_chart_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        band_chart_layout = QVBoxLayout(band_chart_container)
        band_chart_layout.setContentsMargins(8, 8, 8, 8)
        band_chart_layout.setSpacing(0)

        # 创建连续频谱图（统一样式）
        self.band_plot = pg.PlotWidget()
        self.band_plot.setBackground('#14171a')
        self.band_plot.showGrid(x=False, y=False)

        # Y轴自适应
        self.band_plot.enableAutoRange(axis='y', enable=True)
        self.band_plot.getAxis('left').enableAutoSIPrefix(False)

        # X轴固定范围（0-50 Hz）
        self.band_plot.setXRange(0, 50, padding=0)

        # 禁用鼠标拖拽
        self.band_plot.setMouseEnabled(x=False, y=False)

        # 设置坐标轴标签（统一样式）
        self.band_plot.setLabel(
            'left',
            '能量 (dB)',
            **{'color': '#64748b', 'font-size': '11px', 'font-family': 'Segoe UI'}
        )
        self.band_plot.setLabel(
            'bottom',
            '频率 (Hz)',
            **{'color': '#64748b', 'font-size': '11px', 'font-family': 'Segoe UI'}
        )

        self.band_plot.setMinimumHeight(200)

        # 添加频带分割虚线
        band_dividers = [4, 8, 13, 30]
        for freq in band_dividers:
            line = pg.InfiniteLine(pos=freq, angle=90, movable=False)
            line.setPen(pg.mkPen('#3f444e', width=1, style=Qt.DashLine))
            self.band_plot.addItem(line, ignoreBounds=True)

        # 添加频带标记文字
        band_labels = [(2, 'δ'), (6, 'θ'), (10.5, 'α'), (21.5, 'β'), (40, 'γ')]
        for pos, symbol in band_labels:
            text_item = pg.TextItem(text=symbol, color='#64748b', anchor=(0.5, 1))
            text_item.setFont(QFont('Segoe UI', 12, QFont.Bold))
            text_item.setPos(pos, 15)
            self.band_plot.addItem(text_item, ignoreBounds=True)

        # 定义通道颜色
        channel_colors = [
            '#2ecc71', '#e74c3c', '#3498db', '#f39c12',
            '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6'
        ]

        # 创建8条频谱曲线
        self.fft_curves = []
        for color in channel_colors:
            pen = pg.mkPen(color, width=2)
            curve = self.band_plot.plot(pen=pen)
            self.fft_curves.append(curve)

        # 统一边框风格
        box_pen = pg.mkPen(color='#4a5568', width=1.5)
        self.band_plot.showAxis('top')
        self.band_plot.showAxis('right')
        self.band_plot.getAxis('top').setStyle(showValues=False)
        self.band_plot.getAxis('right').setStyle(showValues=False)
        self.band_plot.getAxis('left').setPen(box_pen)
        self.band_plot.getAxis('bottom').setPen(box_pen)
        self.band_plot.getAxis('top').setPen(box_pen)
        self.band_plot.getAxis('right').setPen(box_pen)
        self.band_plot.getAxis('left').setTextPen(pg.mkPen(color='#64748b', width=1))
        self.band_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#64748b', width=1))

        band_chart_layout.addWidget(self.band_plot)
        band_layout.addWidget(band_chart_container)
        center_layout.addWidget(band_widget, 2)  # 增加stretch，给更多空间

        # ============ 中间下部：情绪分类概率显示 ============
        prob_widget = QWidget()
        prob_widget.setMaximumHeight(380)  # 进一步扩展高度
        prob_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        prob_layout = QVBoxLayout(prob_widget)
        prob_layout.setSpacing(12)
        prob_layout.setContentsMargins(16, 12, 16, 12)

        # 标题（统一风格）
        prob_title = QLabel('情绪分类概率 // PROBABILITY DISTRIBUTION')
        prob_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        prob_layout.addWidget(prob_title)

        # 概率条容器（内层卡片）
        prob_bars_container = QWidget()
        prob_bars_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        prob_bars_layout = QVBoxLayout(prob_bars_container)
        prob_bars_layout.setSpacing(8)
        prob_bars_layout.setContentsMargins(12, 12, 12, 12)

        # 显示层统一为二分类：负性 / 非负性。第三行保留占位，避免改动布局。
        self.prob_bar_neg = self._create_probability_bar_v2("负性", "#dc2626")
        self.prob_bar_neu = self._create_probability_bar_v2("非负性", "#059669")
        self.prob_bar_pos = self._create_probability_bar_v2("--", "#3f444e")
        self.prob_bar_pos.hide()

        prob_bars_layout.addWidget(self.prob_bar_neg)
        prob_bars_layout.addWidget(self.prob_bar_neu)
        prob_bars_layout.addWidget(self.prob_bar_pos)

        prob_layout.addWidget(prob_bars_container)
        center_layout.addWidget(prob_widget, 1)

        # 将center_panel添加到analysis_splitter
        analysis_splitter.addWidget(center_panel)

        # 设置analysis_splitter的比例
        analysis_splitter.setStretchFactor(0, 1)
        analysis_splitter.setStretchFactor(1, 1)
        analysis_splitter.setSizes([1, 1])

        # 将分析面板添加到主工作区（上方占78%，进一步扩展）
        main_workspace_layout.addWidget(analysis_splitter, 7)

        # ============ 主工作区下部：大型8通道时域波形区 ============
        time_domain_widget = QWidget()
        time_domain_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        time_domain_widget.setMaximumHeight(500)  # 降低最大高度限制（缩小1/3）
        time_domain_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        time_domain_layout = QVBoxLayout(time_domain_widget)
        time_domain_layout.setSpacing(8)
        time_domain_layout.setContentsMargins(12, 10, 12, 10)

        # 时域波形标题
        time_title = QLabel('脑电时域信号 (8通道) // TIME DOMAIN SIGNAL')
        time_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        time_domain_layout.addWidget(time_title)

        # 时域绘图区（大容器）
        time_plot_container = QWidget()
        time_plot_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)

        # 创建左右两个子布局，用于分割8通道波形
        self.plot_layout_left = QVBoxLayout()
        self.plot_layout_left.setSpacing(4)
        self.plot_layout_left.setContentsMargins(8, 10, 4, 10)

        self.plot_layout_right = QVBoxLayout()
        self.plot_layout_right.setSpacing(4)
        self.plot_layout_right.setContentsMargins(4, 10, 8, 10)

        # 主布局为水平，包含左右两个子布局
        self.plot_layout = QHBoxLayout()
        self.plot_layout.setSpacing(8)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        # 创建左右两个容器
        plot_container_left = QWidget()
        plot_container_left.setLayout(self.plot_layout_left)
        plot_container_right = QWidget()
        plot_container_right.setLayout(self.plot_layout_right)

        self.plot_layout.addWidget(plot_container_left)
        self.plot_layout.addWidget(plot_container_right)

        time_plot_container.setLayout(self.plot_layout)
        time_domain_layout.addWidget(time_plot_container, 1)

        # 将时域波形区添加到主工作区（下方占22%，缩小1/3）
        main_workspace_layout.addWidget(time_domain_widget, 2)

        # ============ 右侧列：当前情绪状态输出 + 神经刺激控制台 + 系统日志 ============
        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(5, 10, 10, 10)

        # ============ 右侧上部：当前情绪状态输出 ============
        threshold_widget = QWidget()
        threshold_widget.setMaximumHeight(280)  # compact current-state card
        threshold_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        threshold_layout = QVBoxLayout(threshold_widget)
        threshold_layout.setSpacing(12)
        threshold_layout.setContentsMargins(16, 12, 16, 12)

        # 标题（统一风格）
        threshold_title = QLabel('当前情绪状态输出 // CURRENT EMOTION STATE')
        threshold_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        threshold_layout.addWidget(threshold_title)

        # Current state card. The complete negative-score history is generated
        # from predictions_display.csv by a separate plotting script.
        state_container = QWidget()
        state_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
            QLabel {
                border: none;
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        state_layout = QGridLayout(state_container)
        state_layout.setContentsMargins(12, 10, 12, 10)
        state_layout.setHorizontalSpacing(10)
        state_layout.setVerticalSpacing(7)

        def make_state_label(text, value=False):
            label = QLabel(text)
            if value:
                label.setStyleSheet("""
                    color: #e5e7eb;
                    font-size: 13px;
                    font-weight: 600;
                """)
            else:
                label.setStyleSheet("""
                    color: #94a3b8;
                    font-size: 12px;
                """)
            return label

        self.current_state_value_labels = {}
        state_rows = [
            ("trial", "当前 Trial：", "--"),
            ("true_label", "真实标签：", "--"),
            ("display_state", "预测状态：", "--"),
            ("prob_negative", "负性概率：", "--"),
            ("prob_non_negative", "非负性概率：", "--"),
            ("negative_score", "负性得分：", "--"),
            ("prediction_source", "预测来源：", "UDA-DDA 后台预测（DE+LDS, 62ch）"),
        ]
        for row_idx, (key, label_text, default_value) in enumerate(state_rows):
            state_layout.addWidget(make_state_label(label_text), row_idx, 0)
            value_label = make_state_label(default_value, value=True)
            value_label.setWordWrap(True)
            self.current_state_value_labels[key] = value_label
            state_layout.addWidget(value_label, row_idx, 1)

        note_label = QLabel("真实标签仅用于离线验证")
        note_label.setStyleSheet("""
            color: #64748b;
            font-size: 11px;
            font-style: italic;
            border: none;
            background: transparent;
        """)
        state_layout.addWidget(note_label, len(state_rows), 0, 1, 2)
        threshold_layout.addWidget(state_container)

        # 图表容器（内层卡片）
        chart_container = QWidget()
        chart_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.setSpacing(0)

        # 创建pyqtgraph折线图（统一样式）
        self.threshold_plot = pg.PlotWidget()
        self.threshold_plot.setBackground('#14171a')  # 与容器背景一致
        self.threshold_plot.showGrid(x=False, y=False)
        self.threshold_plot.setLabel('left', '负向概率 (%)', **{'color': '#64748b', 'font-size': '11px', 'font-family': 'Segoe UI'})
        self.threshold_plot.setLabel('bottom', '时间', **{'color': '#64748b', 'font-size': '11px', 'font-family': 'Segoe UI'})
        self.threshold_plot.setYRange(0, 100)
        self.threshold_plot.setMaximumHeight(180)
        self.threshold_plot.setMinimumHeight(150)

        # 性能优化
        self.threshold_plot.setDownsampling(auto=True, mode='peak')
        self.threshold_plot.setClipToView(True)
        self.threshold_plot.enableAutoRange(axis='y', enable=False)

        # 情绪得分曲线（科技蓝，低饱和度）
        self.threshold_curve = self.threshold_plot.plot(pen=pg.mkPen('#60a5fa', width=2, antialias=False))
        self.threshold_data = np.zeros(200)

        # 统一边框风格（深灰色系，不要白色边框）
        box_pen = pg.mkPen(color='#4a5568', width=1.5)

        # 显示四个边框
        self.threshold_plot.showAxis('top')
        self.threshold_plot.showAxis('right')
        self.threshold_plot.getAxis('top').setStyle(showValues=False)
        self.threshold_plot.getAxis('right').setStyle(showValues=False)

        # 应用边框样式
        self.threshold_plot.getAxis('left').setPen(box_pen)
        self.threshold_plot.getAxis('bottom').setPen(box_pen)
        self.threshold_plot.getAxis('top').setPen(box_pen)
        self.threshold_plot.getAxis('right').setPen(box_pen)

        # 坐标轴文字颜色
        self.threshold_plot.getAxis('left').setTextPen(pg.mkPen(color='#64748b', width=1))
        self.threshold_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#64748b', width=1))

        chart_layout.addWidget(self.threshold_plot)
        threshold_layout.addWidget(chart_container)
        chart_container.hide()

        # 将当前状态卡片添加到右侧面板
        right_layout.addWidget(threshold_widget)

        # ============ 右侧中部：神经刺激控制台 ============
        stim_widget = QWidget()
        stim_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        stim_layout = QVBoxLayout(stim_widget)
        stim_layout.setSpacing(12)
        stim_layout.setContentsMargins(16, 12, 16, 12)

        # 标题（统一风格）
        stim_title = QLabel('神经刺激器控制台 // STIMULATOR CONTROL')
        stim_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        stim_layout.addWidget(stim_title)

        # 内层容器（控制区 + 波形区）
        stim_inner_layout = QHBoxLayout()
        stim_inner_layout.setSpacing(12)
        stim_inner_layout.setContentsMargins(0, 0, 0, 0)

        # ========== 左侧：控制按钮区 ==========
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(12, 12, 12, 12)

        # 分组标题
        control_group_label = QLabel('串口设置')
        control_group_label.setStyleSheet("""
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """)
        control_layout.addWidget(control_group_label)

        # 串口选择
        serial_layout = QHBoxLayout()
        serial_layout.setSpacing(8)
        self.stim_port_combo = QComboBox()
        self.stim_port_combo.setMinimumWidth(120)
        self.stim_port_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #3f444e;
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QComboBox:hover {
                border: 1px solid #4a5568;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        serial_layout.addWidget(self.stim_port_combo)

        self.refresh_stim_btn = QPushButton('⟳')
        self.refresh_stim_btn.setFixedWidth(32)
        self.refresh_stim_btn.clicked.connect(self.refresh_stim_ports)
        self.refresh_stim_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #94a3b8;
                border: 1px solid #475569;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #475569;
                color: #cbd5e1;
            }
        """)
        serial_layout.addWidget(self.refresh_stim_btn)
        control_layout.addLayout(serial_layout)

        # 连接按钮（次级按钮）
        self.btn_stim_connect = QPushButton('连接刺激器')
        self.btn_stim_connect.clicked.connect(self.connect_stimulator)
        self.btn_stim_connect.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
        """)
        control_layout.addWidget(self.btn_stim_connect)

        # 分隔线
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #2a2e36;")
        control_layout.addWidget(separator)

        # 参数配置分组
        params_label = QLabel('参数配置')
        params_label.setStyleSheet("""
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """)
        control_layout.addWidget(params_label)

        self.config_params_btn = QPushButton('下发刺激参数')
        self.config_params_btn.clicked.connect(self.send_stim_params)
        self.config_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        control_layout.addWidget(self.config_params_btn)

        # 分隔线
        separator2 = QWidget()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet("background-color: #2a2e36;")
        control_layout.addWidget(separator2)

        # 手动控制分组
        manual_label = QLabel('手动控制')
        manual_label.setStyleSheet("""
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """)
        control_layout.addWidget(manual_label)

        # 手动触发按钮（主按钮）
        self.manual_trigger_btn = QPushButton('手动开启刺激')
        self.manual_trigger_btn.clicked.connect(self.manual_trigger_stimulation)
        self.manual_trigger_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34d399, stop:1 #10b981);
            }
        """)
        control_layout.addWidget(self.manual_trigger_btn)

        # 紧急停止按钮（警示按钮，低饱和度）
        self.emergency_stop_btn = QPushButton('紧急停止')
        self.emergency_stop_btn.clicked.connect(self.emergency_stop_stimulation)
        self.emergency_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
            QPushButton:pressed {
                background-color: #991b1b;
            }
        """)
        control_layout.addWidget(self.emergency_stop_btn)

        control_layout.addStretch()
        stim_inner_layout.addWidget(control_panel, 3)

        # ========== 右侧：波形显示区 ==========
        waveform_container = QWidget()
        waveform_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_layout.setContentsMargins(8, 8, 8, 8)
        waveform_layout.setSpacing(0)

        # 创建刺激波形图（统一样式）
        self.stim_plot = pg.PlotWidget()
        self.stim_plot.setBackground('#14171a')
        self.stim_plot.showGrid(x=False, y=False)
        self.stim_plot.setLabel('bottom', '时间 (ms)', **{'color': '#64748b', 'font-size': '10px', 'font-family': 'Segoe UI'})
        self.stim_plot.setLabel('left', '')
        self.stim_plot.setXRange(0, 100, padding=0)
        self.stim_plot.setYRange(-2, 2, padding=0)
        self.stim_plot.setMouseEnabled(x=False, y=False)

        # 统一边框风格
        box_pen = pg.mkPen(color='#4a5568', width=1.5)
        self.stim_plot.showAxis('top')
        self.stim_plot.showAxis('right')
        self.stim_plot.getAxis('top').setStyle(showValues=False)
        self.stim_plot.getAxis('right').setStyle(showValues=False)
        self.stim_plot.getAxis('left').setStyle(showValues=False)
        self.stim_plot.getAxis('left').setPen(box_pen)
        self.stim_plot.getAxis('bottom').setPen(box_pen)
        self.stim_plot.getAxis('top').setPen(box_pen)
        self.stim_plot.getAxis('right').setPen(box_pen)
        self.stim_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#64748b', width=1))

        # 创建待机平直线（科技蓝）
        self.stim_curve = self.stim_plot.plot(pen=pg.mkPen('#60a5fa', width=2))
        self.stim_data = np.zeros(100)
        self.stim_curve.setData(self.stim_data)

        waveform_layout.addWidget(self.stim_plot)
        stim_inner_layout.addWidget(waveform_container, 7)

        stim_layout.addLayout(stim_inner_layout)
        right_layout.addWidget(stim_widget, 1)

        # ============ 右侧下部：系统日志（专业科研级）============
        log_widget = QWidget()
        log_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
                border: 1px solid #2a2e36;
            }
        """)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setSpacing(12)
        log_layout.setContentsMargins(16, 12, 16, 12)

        # 标题（统一风格）
        log_title = QLabel('系统日志 // SYSTEM LOG')
        log_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #60a5fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2e36;
        """)
        log_layout.addWidget(log_title)

        # 日志文本框容器（内层卡片）
        log_container = QWidget()
        log_container.setStyleSheet("""
            QWidget {
                background-color: #14171a;
                border-radius: 6px;
                border: 1px solid #252830;
            }
        """)
        log_container_layout = QVBoxLayout(log_container)
        log_container_layout.setContentsMargins(10, 10, 10, 10)
        log_container_layout.setSpacing(0)

        # 日志文本框（监控终端风格，克制版）
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #94a3b8;
                border: none;
                border-radius: 4px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
            }
            QScrollBar:vertical {
                background-color: #1a1d23;
                width: 6px;
                border-radius: 3px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #3f444e;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4a5568;
            }
        """)
        log_container_layout.addWidget(self.log_text)
        log_layout.addWidget(log_container)
        right_layout.addWidget(log_widget, 1)

        # 将main_workspace和right_panel添加到main_splitter
        main_splitter.addWidget(main_workspace)
        main_splitter.addWidget(right_panel)

        # ============ 设置主分割器比例（主工作区:右侧栏 = 75:25）============
        main_splitter.setStretchFactor(0, 75)
        main_splitter.setStretchFactor(1, 25)
        main_splitter.setSizes([750, 250])
        main_splitter.setChildrenCollapsible(False)

        # 将main_splitter添加到主布局
        main_layout.addWidget(main_splitter, 1)

        # ============ 底部信息栏 ============
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(40)
        bottom_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a0e27, stop:0.5 #1a1a2e, stop:1 #0f0f23);
                border-top: 1px solid #2d3748;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setSpacing(30)
        bottom_layout.setContentsMargins(20, 5, 20, 5)

        # 数据包计数
        self.packet_count_label = QLabel('数据包: 0')
        self.packet_count_label.setStyleSheet('color: #64748b; font-size: 11px;')
        bottom_layout.addWidget(self.packet_count_label)

        # 最后刺激时间
        self.last_stimulus_label = QLabel('最后刺激: --')
        self.last_stimulus_label.setStyleSheet('color: #f59e0b; font-size: 11px; font-weight: bold;')
        bottom_layout.addWidget(self.last_stimulus_label)

        # 调试信息
        self.debug_label = QLabel('系统就绪')
        self.debug_label.setStyleSheet('color: #64748b; font-size: 11px; font-family: Consolas;')
        bottom_layout.addWidget(self.debug_label, 1)

        # Y轴量程选择
        yrange_widget = QWidget()
        yrange_layout = QHBoxLayout(yrange_widget)
        yrange_layout.setSpacing(8)
        yrange_layout.setContentsMargins(0, 0, 0, 0)

        yrange_label = QLabel('量程:')
        yrange_label.setStyleSheet('color: #64748b; font-size: 11px;')
        yrange_layout.addWidget(yrange_label)

        self.yrange_combo = QComboBox()
        self.yrange_combo.addItems(['±50µV', '±100µV', '±250µV', '±500µV', '自动'])
        self.yrange_combo.setCurrentText('自动')
        self.yrange_combo.setMinimumWidth(90)
        self.yrange_combo.currentTextChanged.connect(self.on_yrange_changed)
        self.yrange_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #00d4ff;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
        """)
        yrange_layout.addWidget(self.yrange_combo)

        # 0Ω修正开关
        self.correction_checkbox = QCheckBox('0Ω修正')
        self.correction_checkbox.stateChanged.connect(self.toggle_correction)
        self.correction_checkbox.setStyleSheet("""
            QCheckBox {
                color: #64748b;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #2d3748;
                border-radius: 3px;
                background-color: #1a202c;
            }
            QCheckBox::indicator:checked {
                background-color: #8b5cf6;
                border: 2px solid #8b5cf6;
            }
        """)
        yrange_layout.addWidget(self.correction_checkbox)

        bottom_layout.addWidget(yrange_widget)

        main_layout.addWidget(bottom_bar)

        # ============ 初始化刺激波形为待机状态 ============
        self.clear_stim_display()

        # ============ 刷新刺激器串口列表 ============
        self.refresh_ports()

    def _create_status_indicator(self, text, color):
        """创建状态指示器组件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 发光圆点
        dot = QLabel("●")
        dot.setObjectName("status_dot")
        dot.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 20px;
                background: transparent;
            }}
        """)
        dot.setGraphicsEffect(self._create_glow_effect(color))
        layout.addWidget(dot)

        # 状态文字
        label = QLabel(text)
        label.setObjectName("status_text")
        label.setStyleSheet(f"""
            color: {color};
            font-size: 13px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        """)
        layout.addWidget(label)

        return widget

    def _create_glow_effect(self, color):
        """创建霓虹发光效果"""
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(15)
        effect.setColor(QColor(color))
        effect.setOffset(0, 0)
        return effect

    def _create_probability_bar(self, label_text, color):
        """创建情绪概率显示条

        Args:
            label_text: 类别标签("负向"/"中性"/"正向")
            color: 进度条颜色(十六进制,如"#e74c3c"红色)

        Returns:
            QWidget: 包含标签和进度条的组件
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 5, 0, 5)

        # 类别标签
        label = QLabel(f"{label_text}:")
        label.setFixedWidth(60)
        label.setStyleSheet(f"""
            font-size: 13px;
            font-weight: bold;
            color: {color};
        """)
        layout.addWidget(label)

        # 概率进度条
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")  # 显示百分比
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {color};
                border-radius: 5px;
                text-align: center;
                background-color: #1e293b;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
            }}
            QProgressBar::chunk:horizontal {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress_bar, 1)  # stretch=1

        # 存储进度条引用(用于更新)
        widget.label = label
        widget.progress_bar = progress_bar

        return widget

    def _create_probability_bar_v2(self, label_text, color):
        """创建精致版情绪概率显示条（专业科研级）

        Args:
            label_text: 类别标签("负向"/"中性"/"正向")
            color: 进度条颜色(低饱和度，如"#dc2626"深红色)

        Returns:
            QWidget: 包含标签和进度条的组件
        """
        widget = QWidget()
        widget.setStyleSheet("background-color: transparent;")
        layout = QHBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 4, 0, 4)

        # 类别标签（统一样式）
        label = QLabel(f"{label_text}")
        label.setFixedWidth(50)
        label.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)
        layout.addWidget(label)

        # 概率进度条（精致设计）
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%v%")
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #3f444e;
                border-radius: 4px;
                text-align: center;
                background-color: #1e293b;
                color: #cbd5e1;
                font-weight: 600;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 1px;
            }}
            QProgressBar::chunk:horizontal {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress_bar, 1)  # stretch=1

        # 存储进度条引用(用于更新)
        widget.progress_bar = progress_bar

        return widget

    def _draw_head_contour(self):
        """绘制头部俯视图轮廓（10-20系统标准头部）"""
        import math

        # 头部外轮廓（椭圆形）
        theta = np.linspace(0, 2 * np.pi, 100)
        head_x = 0.95 * np.cos(theta)
        head_y = 0.85 * np.sin(theta) * 0.9  # 略微压扁

        # 绘制头部轮廓
        head_contour = self.eeg_map_plot.plot(
            head_x, head_y,
            pen=pg.mkPen(color='#3f444e', width=2.5, style=Qt.SolidLine)
        )

        # 鼻尖方向标识（顶部小三角形）
        nose_x = [0, -0.08, 0.08, 0]
        nose_y = [0.95, 0.82, 0.82, 0.95]
        self.eeg_map_plot.plot(
            nose_x, nose_y,
            pen=pg.mkPen(color='#60a5fa', width=2),
            fill=pg.mkBrush(color='#60a5fa')
        )

        # 左耳轮廓（简化版）
        left_ear_x = [-0.95, -1.05, -1.05, -0.95]
        left_ear_y = [0.2, 0.2, -0.1, -0.1]
        self.eeg_map_plot.plot(
            left_ear_x, left_ear_y,
            pen=pg.mkPen(color='#475569', width=2)
        )

        # 右耳轮廓（简化版）
        right_ear_x = [0.95, 1.05, 1.05, 0.95]
        right_ear_y = [0.2, 0.2, -0.1, -0.1]
        self.eeg_map_plot.plot(
            right_ear_x, right_ear_y,
            pen=pg.mkPen(color='#475569', width=2)
        )

        # 添加方向标识文本
        nose_text = pg.TextItem(text='N', color='#60a5fa', anchor=(0.5, 0))
        nose_text.setPos(0, 1.02)
        nose_text.setFont(QFont('Arial', 10, QFont.Bold))
        self.eeg_map_plot.addItem(nose_text)

        # 添加耳朵标识
        left_ear_text = pg.TextItem(text='L', color='#64748b', anchor=(0.5, 0.5))
        left_ear_text.setPos(-1.1, 0.05)
        left_ear_text.setFont(QFont('Arial', 9))
        self.eeg_map_plot.addItem(left_ear_text)

        right_ear_text = pg.TextItem(text='R', color='#64748b', anchor=(0.5, 0.5))
        right_ear_text.setPos(1.1, 0.05)
        right_ear_text.setFont(QFont('Arial', 9))
        self.eeg_map_plot.addItem(right_ear_text)

    def on_eeg_map_hover(self, pos):
        """处理鼠标悬停效果"""
        # 将场景坐标转换为视图坐标
        view_pos = self.eeg_map_plot.plotItem.vb.mapSceneToView(pos)
        x, y = view_pos.x(), view_pos.y()

        # 查找最近的电极点
        min_distance = float('inf')
        closest_electrode = None

        for name, data in self.eeg_electrode_items.items():
            ex, ey = data['pos']
            distance = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_electrode = name

        # 更新hover状态
        hover_threshold = 0.12
        needs_update = False

        for name, data in self.eeg_electrode_items.items():
            is_hovered = (name == closest_electrode and min_distance < hover_threshold)
            if data.get('hovered', False) != is_hovered:
                data['hovered'] = is_hovered
                needs_update = True

        # 只在状态变化时更新显示
        if needs_update:
            self._update_eeg_map_display()

    def _update_eeg_map_display(self):
        """更新电极分布图显示（支持四种状态：默认、hover、映射、选中）"""
        spots = []

        for name, data in self.eeg_electrode_items.items():
            x, y = data['pos']
            selected = data['selected']
            hovered = data.get('hovered', False)
            mapped = data.get('mapped', False)

            # 四种状态的视觉效果（优先级：选中 > 映射 > hover > 默认）
            if selected:
                # 选中状态：高亮填充 + 外圈光晕
                color = (96, 165, 250)  # 科技蓝 #60a5fa
                size = 18
                pen = pg.mkPen(color='#60a5fa', width=2.5)
                # 添加光晕效果（通过额外的半透明层）
                brush = pg.mkBrush(color='#60a5fa')
            elif mapped:
                # 映射状态：绿色高亮（表示该电极已映射到物理通道）
                color = (34, 197, 94)  # 绿色 #22c55e
                size = 15
                pen = pg.mkPen(color='#22c55e', width=2)
                brush = pg.mkBrush(color='#22c55e')
            elif hovered:
                # Hover状态：轻微高亮
                color = (148, 163, 184)  # 浅灰色 #94b8d8
                size = 16
                pen = pg.mkPen(color='#94b8d8', width=1.5)
                brush = pg.mkBrush(color='#94b8d8')
            else:
                # 默认状态：精致小圆点 + 细描边
                color = (71, 85, 105)  # 深灰色 #475569
                size = 13
                pen = pg.mkPen(color='#64748b', width=1)
                brush = pg.mkBrush(color='#1e293b')

            spots.append({
                'pos': (x, y),
                'size': size,
                'pen': pen,
                'brush': brush
            })

            # 更新文本标签颜色（根据状态变化）
            if name in self.eeg_text_items:
                text_item = self.eeg_text_items[name]
                if selected:
                    text_item.setColor('#60a5fa')  # 选中：亮蓝
                    text_item.setFont(QFont('Arial', 9, QFont.Bold))
                elif mapped:
                    text_item.setColor('#22c55e')  # 映射：绿色
                    text_item.setFont(QFont('Arial', 8, QFont.Bold))
                elif hovered:
                    text_item.setColor('#cbd5e1')  # hover：浅灰
                    text_item.setFont(QFont('Arial', 8))
                else:
                    text_item.setColor('#6b7280')  # 默认：暗灰
                    text_item.setFont(QFont('Arial', 8))

        self.eeg_electrode_scatter.setData(spots)

    def on_mapping_combo_changed(self, changed_index, changed_combo, new_text):
        """
        下拉框变化回调，防止重复选择同一电极

        Args:
            changed_index: 变化的下拉框索引（0-7）
            changed_combo: 变化的下拉框对象
            new_text: 新选择的文本
        """
        # 允许重复选择"未使用"
        if new_text == '未使用':
            return

        # 检查其他下拉框是否已选择该电极
        for i, combo in enumerate(self.channel_mapping_combos):
            if i != changed_index and combo.currentText() == new_text:
                # 发现重复，恢复为"未使用"
                combo.blockSignals(True)
                combo.setCurrentText('未使用')
                combo.blockSignals(False)
                log_emitter.log(f"电极 {new_text} 已被其他通道使用，已自动取消", "warning")
                break

    def confirm_channel_mapping(self):
        """确认通道映射配置"""
        # 从8个下拉框读取映射关系
        new_mapping = {}
        for i, combo in enumerate(self.channel_mapping_combos):
            physical_ch = self.physical_channels[i]
            electrode_pos = combo.currentText()
            new_mapping[physical_ch] = electrode_pos

        # 更新通道映射
        self.channel_mapping = new_mapping

        # ============ 更新10-20系统图的电极映射状态 ============
        # 首先重置所有电极的mapped状态
        for name in self.eeg_electrode_items:
            self.eeg_electrode_items[name]['mapped'] = False

        # 为映射到的电极设置mapped=True
        mapped_electrodes = []
        for ch, elec in new_mapping.items():
            if elec != '未使用' and elec in self.eeg_electrode_items:
                self.eeg_electrode_items[elec]['mapped'] = True
                mapped_electrodes.append(elec)

        # 刷新10-20系统图显示
        self._update_eeg_map_display()

        # 生成映射信息字符串
        mapping_info = []
        for ch, elec in new_mapping.items():
            if elec != '未使用':
                mapping_info.append(f"{ch}→{elec}")

        log_emitter.log(f"通道映射已更新: {', '.join(mapping_info)}", "success")

        # 如果已经有通道显示，重新创建以应用新映射
        if self.num_channels > 0:
            # 清除旧的时域图
            for plot in self.plots:
                plot.setParent(None)
            self.plots.clear()
            self.curves.clear()
            self.data_buffers.clear()
            self.filtered_buffers.clear()
            self.full_data_buffers.clear()
            self.full_filtered_buffers.clear()
            self.input_num_channels = 0

            # 重新创建
            self.create_channels(self.num_channels)

            log_emitter.log(f"已应用新映射，重新创建 {self.num_channels} 个通道显示", "info")

        # 保存映射关系到MAT会话
        if self.mat_session_active:
            self.session_manager.append_event(
                'channel_mapping_changed',
                'MAPPING',
                f'Channel mapping: {", ".join(mapping_info)}'
            )

            # 更新通道信息
            all_channels = list(self.eeg_positions.keys())
            selected_electrodes = [e for e in new_mapping.values() if e != '未使用']
            channel_dict = {
                'selected_channels': selected_electrodes,
                'all_available_channels': all_channels,
                'montage_name': 'International 10-20',
                'reference_channel': 'CMS',
                'ground_channel': 'GND',
                'channel_units': 'µV',
                'physical_channels': self.physical_channels[:len(selected_electrodes)],
                'electrode_mapping': new_mapping
            }
            self.session_manager.update_channel_info(channel_dict)

    def create_channels(self, num_channels):
        """动态创建指定数量的通道（支持最多32通道）"""
        if self.num_channels == num_channels:
            return  # 已经创建过，无需重复创建

        # 清除旧的通道
        for plot in self.plots:
            plot.setParent(None)
        self.plots.clear()
        self.curves.clear()
        self.data_buffers.clear()
        self.filtered_buffers.clear()
        self.full_data_buffers.clear()
        self.full_filtered_buffers.clear()

        self.num_channels = num_channels
        self.display_num_channels = num_channels

        # ============ 按物理通道顺序生成通道名称 ============
        channel_names = []
        for i in range(num_channels):
            physical_ch = self.physical_channels[i]  # CH1, CH2, ...
            electrode_pos = self.channel_mapping.get(physical_ch, '未使用')

            if electrode_pos != '未使用':
                # 格式：电极位置 (CH#)
                display_name = f'{electrode_pos} ({physical_ch})'
            else:
                # 未使用的通道
                display_name = f'{physical_ch} (未使用)'

            channel_names.append(display_name)

        # 更专业的配色方案（支持32个通道）
        colors = [
            '#2ecc71', '#e74c3c', '#3498db', '#f39c12',  # 绿、红、蓝、橙
            '#9b59b6', '#1abc9c', '#e67e22', '#34495e',  # 紫、青、深橙、深蓝
            '#16a085', '#27ae60', '#2980b9', '#8e44ad',  # 深青、深绿、深蓝、深紫
            '#f1c40f', '#e67e22', '#ecf0f1', '#95a5a6',  # 深黄、深橙、白、灰
            '#1abc9c', '#2ecc71', '#3498db', '#9b59b6',  # 重复颜色
            '#e74c3c', '#f39c12', '#16a085', '#27ae60',  # 重复颜色
            '#2980b9', '#8e44ad', '#f1c40f', '#d35400'   # 重复颜色
        ]

        # ============ 创建时域图 ============
        for i in range(num_channels):
            # 创建数据缓冲区（原始数据 + 滤波后数据）
            self.data_buffers.append(np.zeros(1000))
            self.filtered_buffers.append(np.zeros(1000))

            color = colors[i % len(colors)]
            channel_name = channel_names[i] if i < len(channel_names) else f'CH{i+1}'

            # 创建时域图
            plot_widget = pg.PlotWidget(title=f'{channel_name}')
            plot_widget.setYRange(-5000, 5000)
            plot_widget.showGrid(x=False, y=False)

            # 隐藏左侧Y轴刻度
            plot_widget.setLabel('left', '')
            plot_widget.hideAxis('left')  # 完全隐藏左侧Y轴

            plot_widget.setLabel('bottom', '采样点', **{'color': '#bdc3c7', 'font-size': '10px'})

            plot_widget.getAxis('bottom').setStyle(tickFont=QFont('Arial', 9))
            plot_widget.setMinimumHeight(150 if num_channels > 8 else 170)
            plot_widget.setMaximumHeight(170 if num_channels > 8 else 190)

            # 设置标题样式
            plot_widget.setTitle(f'{channel_name}', color='#ecf0f1', size='10pt', bold=True)

            # 创建时域曲线（加粗画笔）
            pen = pg.mkPen(color=color, width=2, style=Qt.SolidLine)
            curve = plot_widget.plot(pen=pen)

            self.plots.append(plot_widget)
            self.curves.append(curve)

            # 将前4个通道添加到左侧布局，后4个通道添加到右侧布局
            if i < 4:
                self.plot_layout_left.addWidget(plot_widget)
            else:
                self.plot_layout_right.addWidget(plot_widget)

        log_emitter.log(f"已创建 {num_channels} 个通道时域图", "info")

    def refresh_ports(self):
        """刷新可用串口列表"""
        # 刷新 EEG 数据串口列表
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if ports:
            for port in ports:
                self.port_combo.addItem(port.device, port.description)
            # 尝试自动选择COM5
            for i in range(self.port_combo.count()):
                if 'COM5' in self.port_combo.itemText(i):
                    self.port_combo.setCurrentIndex(i)
                    break
        else:
            self.port_combo.addItem('无可用串口')

        # 刷新刺激器串口列表
        self.stim_port_combo.clear()
        if ports:
            for port in ports:
                self.stim_port_combo.addItem(port.device, port.description)
        else:
            self.stim_port_combo.addItem('无可用串口')

    def refresh_stim_ports(self):
        """刷新刺激器串口列表"""
        self.stim_port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if ports:
            for port in ports:
                self.stim_port_combo.addItem(port.device, port.description)
            log_emitter.log(f"🔄 刺激器串口列表已刷新，共 {len(ports)} 个可用串口", "info")
        else:
            self.stim_port_combo.addItem('无可用串口')
            log_emitter.log("⚠️ 未检测到可用串口", "warning")

    def toggle_serial(self):
        """打开/关闭串口"""
        if self.serial_thread and self.serial_thread.is_alive():
            # 关闭串口
            self.serial_thread.stop()
            self.serial_thread.join(timeout=1.0)
            self.serial_thread = None
            self.connect_btn.setText('🔌 打开')
            self.pause_btn.setEnabled(False)
            self.debug_label.setText('等待数据...')
        else:
            # 打开串口
            port = self.port_combo.currentText()
            if not port or port == '无可用串口':
                return

            baudrate = int(self.baud_combo.currentText())
            self.serial_thread = SerialReaderThread(port, baudrate)
            self.serial_thread.signals.data_received.connect(self.update_data)
            self.serial_thread.signals.connection_status.connect(self.on_connection_status)
            self.serial_thread.start()
            self.connect_btn.setText('🔌 关闭')

    def toggle_pause(self):
        """暂停/继续绘图"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText('▶')
        else:
            self.pause_btn.setText('⏸')

    # ============ 离线回放控制方法 ============
    def _reset_prediction_sync(self):
        self.current_prediction_index = 0
        self.last_prediction_sync_log_time = 0.0
        self.recent_negative_probs.clear()
        self.decision_window_counter = 0
        self.smoothed_negative_score = None
        if hasattr(self, "current_state_value_labels"):
            defaults = {
                "trial": "--",
                "true_label": "--",
                "display_state": "--",
                "prob_negative": "--",
                "prob_non_negative": "--",
                "negative_score": "--",
                "prediction_source": "UDA-DDA 后台预测（DE+LDS, 62ch）",
            }
            for key, value in defaults.items():
                if key in self.current_state_value_labels:
                    self.current_state_value_labels[key].setText(value)

    def _try_load_prediction_csv(self, playback_path):
        """Load background prediction CSV next to a replay MAT/CSV file."""
        self.prediction_df = []
        self.prediction_times = np.array([], dtype=float)
        self.prediction_mode_enabled = False
        self.prediction_csv_path = ""
        self._reset_prediction_sync()

        mat_dir = os.path.dirname(playback_path)
        candidates = [
            os.path.join(mat_dir, "subject15_trial4_15_predictions_display.csv"),
            os.path.join(mat_dir, "subject15_trial4_15_predictions_display_lds_calibration_feature.csv"),
            os.path.join(mat_dir, "subject15_trial4_15_predictions_display_lds_match_training_test.csv"),
            os.path.join(mat_dir, "subject15_trial4_15_predictions.csv"),
            os.path.join(mat_dir, "subject15_trial4_15_predictions_lds_calibration_feature.csv"),
            os.path.join(mat_dir, "subject15_trial4_15_predictions_lds_match_training_test.csv"),
        ]

        prediction_path = None
        for path in candidates:
            exists = os.path.exists(path)
            log_emitter.log(f"[预测同步] 检查预测文件: {path} -> exists={exists}", "info")
            if exists and prediction_path is None:
                prediction_path = path

        if prediction_path is None:
            log_emitter.log(f"未找到 prediction CSV。当前搜索目录: {mat_dir}", "warning")
            log_emitter.log("候选路径:", "warning")
            for path in candidates:
                log_emitter.log(f"  {path}", "warning")
            log_emitter.log("请先运行 generate_upper_demo_predictions_lds.py", "warning")
            return False

        rows = []
        with open(prediction_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "time_sec" not in row:
                    continue
                try:
                    row["_time_sec_float"] = float(row["time_sec"])
                except (TypeError, ValueError):
                    continue
                rows.append(row)

        if not rows:
            log_emitter.log(f"预测结果文件为空或缺少 time_sec: {prediction_path}", "warning")
            return False

        rows.sort(key=lambda r: r["_time_sec_float"])
        self.prediction_df = rows
        self.prediction_times = np.asarray([r["_time_sec_float"] for r in rows], dtype=float)
        self.prediction_mode_enabled = True
        self.prediction_csv_path = prediction_path
        self.current_prediction_index = 0
        log_emitter.log(f"成功加载预测结果文件: {os.path.basename(prediction_path)}", "success")
        log_emitter.log("预测同步模式已启用：上位机仅展示后台预测结果，不实时运行模型推理。", "success")
        log_emitter.log("MAT 用于波形回放；预测状态来自 predictions_display.csv；真实标签仅用于验证日志。", "info")
        return True

    @staticmethod
    def _prediction_float(row, key, default=0.0):
        try:
            value = row.get(key, default)
            if value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _sync_prediction_display(self, current_time_sec):
        """Synchronize GUI state from prediction CSV by current replay time."""
        if not self.prediction_mode_enabled or len(self.prediction_df) == 0:
            return

        idx = int(np.searchsorted(self.prediction_times, current_time_sec, side="right") - 1)
        if idx < 0:
            return
        idx = min(idx, len(self.prediction_df) - 1)
        if idx == self.current_prediction_index and self.decision_window_counter > 0:
            return

        self.current_prediction_index = idx
        row = self.prediction_df[idx]

        p_negative = self._prediction_float(
            row, "display_prob_negative",
            self._prediction_float(row, "prob_negative", 0.0)
        )
        p_non_negative = self._prediction_float(
            row, "display_prob_non_negative",
            self._prediction_float(row, "prob_non_negative", 1.0 - p_negative)
        )
        prob_sum = p_negative + p_non_negative
        if prob_sum > 0:
            p_negative /= prob_sum
            p_non_negative /= prob_sum
        else:
            p_negative, p_non_negative = 0.5, 0.5

        negative_score = self._prediction_float(row, "display_negative_score", p_negative * 100.0)
        display_state = row.get("display_state") or row.get("pred_binary_label") or (
            "负性" if p_negative >= self.negative_threshold else "非负性"
        )
        display_class_idx = 0 if display_state == "负性" else 1

        self.decision_window_counter += 1
        self.prob_bar_neg.progress_bar.setValue(int(round(p_negative * 100)))
        self.prob_bar_neu.progress_bar.setValue(int(round(p_non_negative * 100)))
        self.fsm_status_label.setText(display_state)
        self.fsm_status_label.setStyleSheet("""
            color: #ef4444;
            font-weight: bold;
        """ if display_state == "负性" else """
            color: #10b981;
            font-weight: bold;
        """)

        self.threshold_data[:-1] = self.threshold_data[1:]
        self.threshold_data[-1] = negative_score
        if self.threshold_curve is not None:
            self.threshold_curve.setData(self.threshold_data)

        trial_id = row.get("trial_id", "")
        true_label = row.get("true_label_name", "")
        prediction_source = (
            row.get("display_probability_source")
            or row.get("feature_source")
            or row.get("scaler_mode")
            or "UDA-DDA 后台预测（DE+LDS, 62ch）"
        )
        if hasattr(self, "current_state_value_labels"):
            self.current_state_value_labels["trial"].setText(f"T{trial_id}" if trial_id != "" else "--")
            self.current_state_value_labels["true_label"].setText(str(true_label) if true_label != "" else "--")
            self.current_state_value_labels["display_state"].setText(str(display_state))
            self.current_state_value_labels["prob_negative"].setText(f"{p_negative:.3f}")
            self.current_state_value_labels["prob_non_negative"].setText(f"{p_non_negative:.3f}")
            self.current_state_value_labels["negative_score"].setText(f"{negative_score:.1f} %")
            self.current_state_value_labels["prediction_source"].setText(str(prediction_source))

        if self.is_playback_mode:
            self.online_predictions.append(display_class_idx)

        now = time.time()
        if now - self.last_prediction_sync_log_time >= 1.0:
            log_emitter.log(
                f"[预测同步] t={current_time_sec:.1f}s | trial={trial_id} | "
                f"真实标签={true_label} | 显示状态={display_state} | "
                f"负性得分={negative_score:.1f} | 负性概率={p_negative:.3f} | "
                f"非负性概率={p_non_negative:.3f}",
                "info"
            )
            self.last_prediction_sync_log_time = now

    def load_csv_file(self):
        """加载CSV文件"""
        # 先停止当前的数据源（串口或回放）
        self.stop_current_data_source()

        # 打开文件选择对话框
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            '选择数据文件',
            os.path.join(os.getcwd(), 'data'),
            'MAT文件 (*.mat);;CSV文件 (*.csv);;所有文件 (*)'
        )

        if not filepath:
            return

        # 创建回放线程
        self.playback_thread = OfflinePlaybackThread()
        success, num_channels, num_rows = self.playback_thread.load_data(filepath)

        if success:
            self.playback_file_path = filepath
            self.playback_total_rows = num_rows
            self.is_playback_mode = True
            self._try_load_prediction_csv(filepath)

            # 更新UI
            filename = os.path.basename(filepath)
            self.file_info_label.setText(f'{filename} ({num_rows}行, {num_channels}通道)')
            self.file_info_label.setStyleSheet('font-size: 11px; color: #27ae60;')

            # ============ 解析文件名确定真实标签（离线评测）============
            filename_lower = filename.lower()
            if 'negative' in filename_lower:
                self.current_true_label = 0  # 负向情绪
                log_emitter.log(f"[评测] 文件名包含 'negative'，真实标签设为 0（负向）", "info")
            elif 'neutral' in filename_lower:
                self.current_true_label = 1  # 中性情绪
                log_emitter.log(f"[评测] 文件名包含 'neutral'，真实标签设为 1（中性）", "info")
            elif 'positive' in filename_lower:
                self.current_true_label = 2  # 正向情绪
                log_emitter.log(f"[评测] 文件名包含 'positive'，真实标签设为 2（正向）", "info")
            else:
                self.current_true_label = None
                log_emitter.log(f"[评测] 文件名不包含情绪标签关键词，跳过评测", "warning")

            self.play_pause_btn.setEnabled(True)
            self.stop_playback_btn.setEnabled(True)
            self.play_pause_btn.setText('▶ 播放')

            # 重置进度条和滑块
            self.playback_progress.setValue(0)
            self.playback_slider.setValue(0)

            # 禁用串口控制
            self.connect_btn.setEnabled(False)
            self.port_combo.setEnabled(False)
            self.baud_combo.setEnabled(False)

            log_emitter.log(f"离线回放模式: {filename}", "info")
            self.status_label.setText('● 回放模式')
            self.status_label.setStyleSheet('color: #9b59b6; font-size: 12px; font-weight: bold;')
        else:
            self.playback_thread = None
            self.is_playback_mode = False

    def toggle_playback(self):
        """播放/暂停回放"""
        if not self.playback_thread:
            return

        if self.playback_thread.paused or not self.playback_thread.is_alive():
            # 开始播放
            if not self.playback_thread.is_alive():
                # 首次播放
                self.playback_thread.signals.data_received.connect(self.update_data)
                self.playback_thread.signals.connection_status.connect(self.on_playback_status)
                self.playback_thread.start()

                # ============ 清空预测结果列表（开始新的评测）============
                self.online_predictions = []
                self.smoothed_negative_score = None
                self.decision_window_counter = 0
                self.recent_negative_probs.clear()
                self.current_prediction_index = 0
                self.last_prediction_sync_log_time = 0.0
                self.last_inference_submit_time = 0.0
                self.last_inference_skip_log_time = 0.0
                self.recent_scores = []
                self.recent_emotions = []
                log_emitter.log("开始离线回放，预测列表已清空", "info")

                log_emitter.log("开始离线回放", "info")
            else:
                # 恢复播放
                self.playback_thread.resume()
                log_emitter.log("继续回放", "info")

            self.play_pause_btn.setText('⏸ 暂停')
            self.play_pause_btn.setStyleSheet('background-color: #f39c12;')
        else:
            # 暂停播放
            self.playback_thread.pause()
            self.play_pause_btn.setText('▶ 继续')
            self.play_pause_btn.setStyleSheet('')

            # 打印阶段性评测报告
            if self.is_playback_mode and self.current_true_label is not None:
                self.print_evaluation_report(title="离线回放阶段性评测报告（暂停）")

            log_emitter.log("暂停回放", "info")

    def stop_playback(self):
        """停止回放"""
        if not self.playback_thread:
            return

        # 打印阶段性评测报告（在停止前）
        if self.is_playback_mode and self.current_true_label is not None:
            self.print_evaluation_report(title="离线回放阶段性评测报告（停止）")

        self.playback_thread.stop()
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

        # 重置UI
        self.play_pause_btn.setText('▶ 播放')
        self.play_pause_btn.setStyleSheet('')
        self.play_pause_btn.setEnabled(False)
        self.stop_playback_btn.setEnabled(False)
        self.playback_progress.setValue(0)
        self.playback_slider.setValue(0)

        # 重新启用串口控制
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)

        self.playback_thread = None
        self.is_playback_mode = False
        self.prediction_mode_enabled = False
        self.smoothed_negative_score = None
        self.decision_window_counter = 0
        self.recent_negative_probs.clear()
        self.current_prediction_index = 0
        self.last_prediction_sync_log_time = 0.0
        self.last_inference_submit_time = 0.0
        self.last_inference_skip_log_time = 0.0
        self.recent_scores = []
        self.recent_emotions = []
        self.status_label.setText('● 未连接')
        self.status_label.setStyleSheet('color: #e74c3c; font-size: 12px; font-weight: bold;')
        log_emitter.log("回放已停止", "warning")

    def on_playback_speed_changed(self, text):
        """播放速度变化"""
        if not self.playback_thread:
            return

        # 提取数字
        import re
        match = re.search(r'([\d.]+)', text)
        if match:
            speed = float(match.group(1))
            self.playback_thread.set_playback_speed(speed)
            log_emitter.log(f"播放速度: {text}", "info")

    def on_playback_status(self, connected, message):
        """处理回放状态变化"""
        if not connected:
            # 回放结束
            self.play_pause_btn.setText('▶ 播放')
            self.play_pause_btn.setStyleSheet('')

            # ============ 生成离线回放评测报告 ============
            if self.is_playback_mode and self.current_true_label is not None and len(self.online_predictions) > 0:
                self.generate_evaluation_report()

        elif '进度:' in message:
            # 更新进度
            try:
                progress = int(message.split('进度:')[1].split('%')[0])
                self.playback_progress.setValue(progress)
                if not self.playback_slider.isSliderDown():
                    self.playback_slider.blockSignals(True)
                    self.playback_slider.setValue(progress)
                    self.playback_slider.blockSignals(False)
            except:
                pass

    def on_slider_pressed(self):
        """滑块按下（暂停自动更新）"""
        pass

    def on_slider_released(self):
        """滑块释放（跳转到指定位置）"""
        if not self.playback_thread:
            return

        progress = self.playback_slider.value()
        target_index = int(progress / 100 * self.playback_total_rows)
        self.playback_thread.set_position(target_index)
        log_emitter.log(f"跳转到: {progress}%", "info")

    def on_slider_changed(self, value):
        """滑块值变化（仅更新显示）"""
        pass

    def stop_current_data_source(self):
        """停止当前数据源（串口或回放）"""
        # 停止串口
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.stop()
            self.serial_thread.join(timeout=1.0)
            self.serial_thread = None
            self.connect_btn.setText('🔌 打开')
            self.pause_btn.setEnabled(False)

        # 停止回放
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.stop()
            self.playback_thread.join(timeout=1.0)
            self.playback_thread = None
            self.play_pause_btn.setEnabled(False)
            self.stop_playback_btn.setEnabled(False)

    def toggle_correction(self, state):
        """切换0Ω修正开关"""
        self.zero_ohm_correction = (state == Qt.Checked)
        correction_mode = "开启" if self.zero_ohm_correction else "关闭"
        log_emitter.log(f"0Ω修正已{correction_mode}（{'信号发生器测试模式' if self.zero_ohm_correction else '正常脑电模式'}）", "info")

    def on_yrange_changed(self, text):
        """Y轴量程下拉框变化处理"""
        # 解析量程值
        if '自动' in text:
            self.y_axis_range = 'auto'
            log_emitter.log(f"Y轴量程已设置为：自动缩放", "info")
        else:
            # 提取数字（例如从"±50µV"中提取"50"）
            import re
            match = re.search(r'(\d+)', text)
            if match:
                range_value = match.group(1)
                self.y_axis_range = range_value
                limit = self.y_range_limits[range_value]
                log_emitter.log(f"Y轴量程已设置为：±{limit}µV", "info")

                # 立即应用新的Y轴范围到所有通道
                if self.plots:
                    for plot in self.plots:
                        plot.setYRange(-limit, limit, padding=0)

    def on_model_changed(self, text):
        """Handle prediction source selection changes."""
        model_type = self.model_selector_combo.currentData() or PREDICTION_SYNC_SOURCE_TYPE
        model_name = self.model_selector_combo.currentText()

        if model_type == PREDICTION_SYNC_SOURCE_TYPE:
            log_emitter.log(f"正在切换预测来源: {model_name} ({model_type})", "info")
            if hasattr(self, "model_thread"):
                self.model_thread.set_model(model_type)
            log_emitter.log("预测同步来源已启用：不会加载或实时运行 SVM/EEGNet/UDA-DDA。", "success")
            return

        if model_type == 'uda-dda':
            model_type = 'uda-dda-online'
        if hasattr(self, 'model_thread'):
            log_emitter.log(f"正在切换预测模型: {model_name} ({model_type})", "info")
            self.model_thread.set_model(model_type)
            log_emitter.log(f"已切换模型: {model_type}", "success")

    def toggle_recording(self):
        """切换数据记录状态（MAT会话保存）"""
        if not self.is_recording:
            # ============ 启动MAT会话 ============
            import time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            meta_info = {
                'subject_id': 'SUB01',  # 可配置：从UI获取
                'notes': f'EEG session recorded at {timestamp}'
            }
            config_info = {
                'model_name': self.model_selector_combo.currentData(),
                'sampling_rate': 200,
                'channel_count': self.num_channels,
                'eeg_serial_port': self.port_selector_combo.currentText(),
                'eeg_baudrate': 460800,
                'stim_serial_port': self.stim_port_selector.currentText() if hasattr(self, 'stim_port_selector') else '',
                'stim_baudrate': 115200
            }

            self.session_manager.start_session(meta_info, config_info)
            self.mat_session_active = True

            # 记录通道信息
            if hasattr(self, 'eeg_electrode_items'):
                selected_channels = [name for name, data in self.eeg_electrode_items.items() if data.get('selected', False)]
                all_channels = list(self.eeg_electrode_items.keys())

                channel_dict = {
                    'selected_channels': selected_channels,
                    'all_available_channels': all_channels,
                    'montage_name': 'International 10-20',
                    'reference_channel': 'CMS',
                    'ground_channel': 'GND',
                    'channel_units': 'µV'
                }
                self.session_manager.update_channel_info(channel_dict)

            # 更新UI状态
            self.is_recording = True
            self.recording_start_time = datetime.now()
            self.record_btn.setText('⏹ 停止')
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc2626;
                    color: #ffffff;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton:hover {
                    background-color: #b91c1c;
                }
            """)

            log_emitter.log(f"MAT会话已启动: {self.session_manager.session_id}", "success")
            self.debug_label.setText(f'MAT会话记录中: {self.session_manager.session_id}')
        else:
            # ============ 结束MAT会话 ============
            if self.mat_session_active:
                saved_path = self.session_manager.end_session()
                self.mat_session_active = False

                if saved_path:
                    log_emitter.log(f"MAT会话已保存: {saved_path}", "success")
                    self.debug_label.setText(f'已保存: {os.path.basename(saved_path)}')
                else:
                    log_emitter.log("MAT会话保存失败", "error")
                    self.debug_label.setText('保存失败')

            # 恢复UI状态
            self.is_recording = False
            self.record_btn.setText('🔴 记录')
            self.record_btn.setStyleSheet('')  # 恢复默认样式

    def start_recording(self, filepath):
        """开始数据记录"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 创建CSV文件并写入表头
            self.csv_file = open(filepath, 'w', newline='', encoding='utf-8-sig')
            self.csv_writer = csv.writer(self.csv_file)

            # 写入表头（需要知道通道数量）
            if self.num_channels > 0:
                headers = ['Time'] + [f'CH{i+1}' for i in range(self.num_channels)]
                self.csv_writer.writerow(headers)
            else:
                # 如果还没有数据，先写入Time，等有数据时再补充通道
                self.csv_writer.writerow(['Time'])

            self.csv_file.flush()  # 确保数据写入磁盘

            self.is_recording = True
            self.recording_start_time = datetime.now()
            self.record_btn.setText('⏹ 停止')
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc2626;
                    color: #ffffff;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton:hover {
                    background-color: #b91c1c;
                }
            """)

            log_emitter.log(f"开始记录数据到: {filepath}", "success")
            self.debug_label.setText(f"记录中: {os.path.basename(filepath)}")

        except Exception as e:
            log_emitter.log(f"创建文件失败: {e}", "error")
            self.debug_label.setText(f"记录失败: {str(e)}")

    def stop_recording(self):
        """停止数据记录"""
        if self.csv_file:
            try:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None

                # 计算记录时长
                if self.recording_start_time:
                    duration = datetime.now() - self.recording_start_time
                    duration_str = str(duration).split('.')[0]  # 去掉微秒
                    log_emitter.log(f"数据记录已停止，时长: {duration_str}", "info")

            except Exception as e:
                log_emitter.log(f"关闭文件时出错: {e}", "error")

        self.is_recording = False
        self.record_btn.setText('🔴 记录')
        self.record_btn.setStyleSheet('')  # 恢复默认样式
        self.debug_label.setText('记录已停止')

    # ============ 决策引擎结果处理 ============
    def handle_decision_result(self, result):
        """
        处理决策引擎结果（简化版，无FSM）
        在主线程中调用，确保线程安全

        Args:
            result: 决策引擎返回的完整字典
        """
        state = result.get('state', 'DIRECT')
        probabilities = result.get('probabilities', None)  # 新增：概率数组
        trigger = result.get('trigger', False)
        reason = result.get('reason', '')
        if self.decision_window_counter < 5 or self.decision_window_counter % 20 == 0:
            log_emitter.log(
                f"handle_decision_result received: score={result.get('score')}, "
                f"ui_emotion={result.get('ui_emotion')}, class_idx={result.get('class_idx')}, "
                f"probabilities={np.asarray(probabilities).reshape(-1).tolist() if probabilities is not None else None}",
                "model"
            )

        if probabilities is None:
            log_emitter.log("模型结果缺少 probabilities，跳过本次显示更新。", "warning")
            return

        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if len(probs) == 3:
            p_negative = float(probs[0])
            p_non_negative = float(probs[1] + probs[2])
            raw_prob_text = np.array2string(probs, precision=3, separator=", ")
        elif len(probs) == 2:
            p_negative = float(probs[0])
            p_non_negative = float(probs[1])
            raw_prob_text = ""
        else:
            log_emitter.log(f"模型概率长度异常: len={len(probs)}，已跳过显示更新。", "warning")
            return

        p_negative = float(np.clip(p_negative, 0.0, 1.0))
        p_non_negative = float(np.clip(p_non_negative, 0.0, 1.0))
        prob_sum = p_negative + p_non_negative
        if prob_sum > 0:
            p_negative /= prob_sum
            p_non_negative /= prob_sum
        else:
            p_negative, p_non_negative = 0.5, 0.5

        raw_negative_score = p_negative * 100.0
        if self.smoothed_negative_score is None:
            self.smoothed_negative_score = raw_negative_score
        else:
            alpha = self.negative_score_ema_alpha
            self.smoothed_negative_score = alpha * raw_negative_score + (1.0 - alpha) * self.smoothed_negative_score

        self.recent_negative_probs.append(p_negative)
        p_negative_display = float(np.mean(self.recent_negative_probs))
        p_non_negative_display = 1.0 - p_negative_display
        negative_score_display = p_negative_display * 100.0
        display_emotion = "负性" if p_negative_display >= self.negative_threshold else "非负性"
        display_class_idx = 0 if display_emotion == "负性" else 1
        self.decision_window_counter += 1

        # ============ 追加预测数据到MAT会话 ============
        if self.mat_session_active and probabilities is not None:
            import time
            pred_dict = {
                'timestamp': time.time(),
                'emotion_label': display_emotion,
                'emotion_id': display_class_idx,
                'probability': list(probabilities) if isinstance(probabilities, (list, np.ndarray)) else [0, 0, 0],
                'score': float(negative_score_display)
            }
            self.session_manager.append_prediction(pred_dict)

        # ============ 收集预测结果（离线回放评测）============
        if self.is_playback_mode and self.current_true_label is not None:
            # 将预测类别追加到列表
            self.online_predictions.append(display_class_idx)

        should_update_ui = (
            self.decision_window_counter % self.ui_decision_update_every_n_windows == 0
        )
        if not should_update_ui and not trigger:
            return

        # ============ 二分类概率显示 ============
        self.prob_bar_neg.progress_bar.setValue(int(round(p_negative_display * 100)))
        self.prob_bar_neu.progress_bar.setValue(int(round(p_non_negative_display * 100)))

        # ============ 系统日志降频与降噪 ============
        # 收集最近的得分和情绪
        self.recent_scores.append(negative_score_display)
        self.recent_emotions.append(display_emotion)

        # 保持列表长度不超过降频因子
        if len(self.recent_scores) > self.ui_decision_update_every_n_windows:
            self.recent_scores.pop(0)
            self.recent_emotions.pop(0)

        # 更新状态标签显示（简化版）
        self.update_state_display(state, '100%', 0, display_emotion)

        # 更新动态阈值折线图（显示短窗口聚合后的负性得分历史）
        self.threshold_data[:-1] = self.threshold_data[1:]
        self.threshold_data[-1] = negative_score_display
        self.threshold_curve.setData(self.threshold_data)

        # 判断是否触发刺激（手动触发时不触发）
        if trigger:
            # 发送UDP刺激命令（关键事件，必须立即打印）
            success = self.udp_sender.send_stimulus("STIM_ON")

            if success:
                # 更新最后刺激时间
                stim_time = datetime.now().strftime('%H:%M:%S')
                self.last_stimulus_label.setText(f'最后刺激: {stim_time}')

                # 用红色高亮打印触发日志（包含情绪标签）
                log_emitter.log(f"🔥 {reason}", "stimulus")

                # 更新刺激次数
                self.stimulus_count += 1

                # ============ 触发刺激波形显示 ============
                self.trigger_stim_display()

                # ============ 发送真实刺激器报文（闭环融合）============
                stim_cmd = "FE FF 01 F2 01 01 00 F5 FE FF"
                self.send_hex_cmd(stim_cmd)
            else:
                log_emitter.log(f"UDP发送失败: {reason}", "error")
        else:
            debug_text = f" | 原始三分类概率: {raw_prob_text}" if raw_prob_text else ""
            score_label = "窗口负性得分" if self.display_decision_window == 1 else "短窗负性得分"
            log_emitter.log(
                f"[监测摘要] 状态: {display_emotion} | {score_label}: {negative_score_display:.1f} "
                f"| 负性概率: {p_negative_display:.3f} | 非负性概率: {p_non_negative_display:.3f}{debug_text}",
                "info"
            )

    def generate_evaluation_report(self):
        """
        生成离线回放评测报告（回放自然结束时调用）

        计算在线预测的准确率，并使用醒目的颜色打印到系统日志
        打印后会清空预测列表，为下一次回放做准备
        """
        # 调用通用打印方法，使用"完整"作为标题
        self.print_evaluation_report(title="离线回放完整评测报告")

        # 清空预测结果列表（为下一次回放做准备）
        self.online_predictions = []

    def print_evaluation_report(self, title="离线回放阶段性评测报告"):
        """
        打印评测报告（用于暂停、停止或结束时的阶段性准确率统计）

        Args:
            title: 报告标题（区分是完整报告还是阶段性报告）

        注意：
            此方法只计算并打印当前的准确率，不会清空 self.online_predictions 列表
        """
        if not self.is_playback_mode or self.current_true_label is None:
            return

        if len(self.online_predictions) == 0:
            log_emitter.log("暂无预测结果，无法生成评测报告", "warning")
            return

        try:
            from sklearn.metrics import accuracy_score, confusion_matrix
            import os

            # 获取真实标签和预测标签
            true_labels = [self.current_true_label] * len(self.online_predictions)
            pred_labels = self.online_predictions

            # 计算准确率
            accuracy = accuracy_score(true_labels, pred_labels)

            # 计算混淆矩阵
            cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

            # 统计各类别的预测次数
            label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            true_label_name = label_names.get(self.current_true_label, 'Unknown')

            # 统计预测分布
            unique, counts = np.unique(pred_labels, return_counts=True)
            pred_distribution = {label_names.get(int(u), f'Class{int(u)}'): int(c) for u, c in zip(unique, counts)}

            # ============ 打印评测报告到 UI 日志 ============
            separator = "=" * 60
            log_emitter.log(separator, "success")
            log_emitter.log(f"📊 {title}", "success")
            log_emitter.log(separator, "success")
            log_emitter.log(f"真实标签: {true_label_name}", "success")
            log_emitter.log(f"预测次数: {len(self.online_predictions)} 次", "success")
            log_emitter.log(f"在线准确率: {accuracy * 100:.2f}%", "success")
            log_emitter.log("-" * 60, "success")

            # 打印预测分布
            log_emitter.log("预测类别分布:", "success")
            for label, count in pred_distribution.items():
                log_emitter.log(f"  - {label}: {count} 次", "success")

            log_emitter.log("-" * 60, "success")

            # 打印混淆矩阵
            log_emitter.log("混淆矩阵:", "success")
            log_emitter.log("       Pred\\True    Neg  Neu  Pos", "success")
            for i, row in enumerate(cm):
                label = ['Neg', 'Neu', 'Pos'][i]
                log_emitter.log(f"       {label:12s} {str(row)}", "success")

            log_emitter.log(separator, "success")

            # 如果准确率较低，给出提示
            if accuracy < 0.6:
                log_emitter.log(f"⚠️  准确率较低 ({accuracy*100:.1f}%)，可能需要检查模型或特征提取参数", "warning")
            elif accuracy >= 0.8:
                log_emitter.log(f"✅ 准确率良好 ({accuracy*100:.1f}%)！", "success")

            # ============ 自动保存评测报告到本地文件 ============
            # 确保目录存在
            report_dir = os.path.join(os.getcwd(), "outputs", "eval_reports")
            os.makedirs(report_dir, exist_ok=True)

            # 生成带时间戳的文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(report_dir, f"EvalReport_{true_label_name}_{timestamp}.txt")

            # 构建报告内容（工整格式）
            report_lines = [
                "=" * 60,
                f"{title}",
                "=" * 60,
                f"真实标签: {true_label_name}",
                f"预测次数: {len(self.online_predictions)} 次",
                f"在线准确率: {accuracy * 100:.2f}%",
                "-" * 60,
                "预测类别分布:",
            ]
            # 添加预测类别分布
            for label, count in pred_distribution.items():
                report_lines.append(f"  - {label}: {count} 次")

            report_lines.extend([
                "-" * 60,
                "混淆矩阵:",
                "       Pred\\True    Neg  Neu  Pos",
            ])
            # 添加混淆矩阵
            for i, row in enumerate(cm):
                label = ['Neg', 'Neu', 'Pos'][i]
                report_lines.append(f"       {label:12s} {str(row)}")

            report_lines.append("=" * 60)

            # 写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            log_emitter.log(f"评测报告已保存到: {filename}", "success")

        except Exception as e:
            log_emitter.log(f"打印评测报告失败: {str(e)}", "error")

    def update_state_display(self, state, progress_info, refractory_remaining, ui_emotion='Unknown'):
        """
        更新状态显示（简化版，已移除状态显示面板）

        Args:
            state: 当前状态（忽略）
            progress_info: 进度信息（忽略）
            refractory_remaining: 不应期剩余次数（忽略）
            ui_emotion: 当前情绪标签（忽略，概率条已显示）
        """
        # 状态监控面板已移除，概率显示直接在概率条中更新
        # 此方法保留为兼容性，不执行任何操作
        pass

    # ============ 刺激波形监控相关 ============
    def trigger_stim_display(self):
        """
        触发刺激波形显示

        生成标准的双相脉冲序列（Biphasic Pulse Train）：
        - 5 个周期的方波
        - 先正后负（阴极-阳极双相刺激）
        - 中间夹杂平直基线
        """
        import time

        # 生成双相脉冲序列数据
        # 参数设置
        num_points = 100  # 总点数
        pulses = 5  # 脉冲数量
        pulse_width = 4  # 每个相位的宽度（点数）
        inter_pulse_gap = 10  # 脉冲间隔（点数）
        amplitude = 1.5  # 脉冲幅度（mA）

        # 初始化波形数据（全零基线）
        waveform = np.zeros(num_points)

        # 生成双相脉冲
        for i in range(pulses):
            # 计算脉冲起始位置
            start_pos = i * (2 * pulse_width + inter_pulse_gap) + 5

            # 确保不超出数据范围
            if start_pos + 2 * pulse_width < num_points:
                # 阴极相（负相，Cathodic）
                waveform[start_pos:start_pos + pulse_width] = -amplitude
                # 阳极相（正相，Anodic）
                waveform[start_pos + pulse_width:start_pos + 2 * pulse_width] = amplitude

        # 更新波形显示（红色加粗）
        self.stim_curve.setData(waveform)
        self.stim_curve.setPen(pg.mkPen(color='r', width=3))

        # 2 秒后自动清理波形，回到待机状态
        QTimer.singleShot(2000, self.clear_stim_display)

    def clear_stim_display(self):
        """
        清理刺激波形，显示待机状态

        画出一条贯穿 X 轴的平直 0V 绿线，代表待机无输出状态
        """
        # 创建全零数据（0V 基线）
        standby_data = np.zeros(100)

        # 更新波形显示（绿色平直线）
        self.stim_curve.setData(standby_data)
        self.stim_curve.setPen(pg.mkPen(color='g', width=2))

    # ============ 刺激器串口通信相关 ============
    def connect_stimulator(self):
        """连接刺激器串口"""
        try:
            # 获取选中的串口
            port_name = self.stim_port_combo.currentText()
            if not port_name:
                log_emitter.log("请先选择刺激器串口", "warning")
                return

            # 如果已经打开，先关闭
            if self.stim_serial is not None and self.stim_serial.is_open:
                self.stim_serial.close()

            # 打开串口（波特率 115200）
            self.stim_serial = serial.Serial(
                port=port_name,
                baudrate=115200,
                timeout=1
            )

            # 更新按钮状态（变绿并显示"已连接"）
            self.btn_stim_connect.setText('已连接')
            self.btn_stim_connect.setEnabled(False)
            self.btn_stim_connect.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: #fff;
                    border: 1px solid #10b981;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)

            log_emitter.log(f"✅ 刺激器已连接: {port_name} @ 115200 baud", "success")

        except Exception as e:
            log_emitter.log(f"❌ 连接刺激器失败: {str(e)}", "error")
            self.stim_serial = None

    def send_hex_cmd(self, hex_string):
        """
        发送十六进制指令到刺激器

        Args:
            hex_string: 十六进制字符串，如 "FE FF 01 F1 01 01..."
        """
        if self.stim_serial is None or not self.stim_serial.is_open:
            log_emitter.log("⚠️ 刺激器未连接，无法发送指令", "warning")
            return False

        try:
            # 去除空格并转换为字节
            hex_data = hex_string.replace(' ', '')
            byte_data = bytes.fromhex(hex_data)

            # 发送指令
            self.stim_serial.write(byte_data)

            # 打印日志
            log_emitter.log(f"➡️ 发送指令: [{hex_string}]", "info")
            return True

        except Exception as e:
            log_emitter.log(f"❌ 发送指令失败: {str(e)}", "error")
            return False

    def send_stim_params(self):
        """下发刺激参数"""
        # 参数配置报文
        cmd = "FE FF 01 F1 01 01 32 00 00 9A 74 00 00 00 FA 00 00 00 FA 01 00 04 29 FE FF"
        self.send_hex_cmd(cmd)

    def manual_trigger_stimulation(self):
        """手动开启刺激"""
        # 发送开启指令
        cmd = "FE FF 01 F2 01 01 00 F5 FE FF"
        success = self.send_hex_cmd(cmd)

        if success:
            # 同时触发波形显示
            self.trigger_stim_display()
            log_emitter.log("🎯 手动触发刺激", "stimulus")

    def emergency_stop_stimulation(self):
        """紧急停止刺激"""
        # 发送停止指令
        cmd = "FE FF 01 F2 01 00 00 F4 FE FF"
        success = self.send_hex_cmd(cmd)

        if success:
            # 立即清理波形显示
            self.clear_stim_display()
            log_emitter.log("🛑 紧急停止刺激", "stimulus")

    def reset_calibration(self):
        """重新标定基线（简化版，只重置得分曲线）"""
        # 重置得分曲线
        self.threshold_data = np.zeros(200)
        self.threshold_curve.setData(self.threshold_data)
        self.smoothed_negative_score = None
        self.decision_window_counter = 0
        self.recent_negative_probs.clear()
        self.last_inference_submit_time = 0.0
        self.last_inference_skip_log_time = 0.0
        self.recent_scores = []
        self.recent_emotions = []

        log_emitter.log("已重置得分历史", "info")

    def on_stimulus_triggered(self, info):
        """刺激触发时的回调（在主线程中执行）"""
        log_emitter.log(f"🚀 {info}", "stimulus")

        # ============ 记录刺激事件到MAT会话 ============
        if self.mat_session_active:
            import time
            stim_dict = {
                'timestamp': time.time(),
                'command': 'STIM_ON',
                'success': True
            }
            self.session_manager.append_stim_record(stim_dict)
            self.session_manager.append_event('stimulus_triggered', 'STIM', info)

    # ============ 日志相关 ============
    def append_log(self, message, color="#ecf0f1"):
        """追加日志（在主线程中执行）"""
        # 获取当前时间
        timestamp = datetime.now().strftime('%H:%M:%S')

        # 构建HTML格式的日志消息
        html = f'<span style="color: #7f8c8d;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'

        # 追加到日志文本框
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html + '<br>')

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        log_emitter.log("日志已清空", "info")

    # ============ 数据更新相关 ============
    def on_connection_status(self, connected, message):
        """处理连接状态变化"""
        if connected:
            # 更新连接状态指示器（绿色）
            self.conn_status_label.setText("已连接")
            self.conn_status_label.setStyleSheet("""
                color: #10b981;
                font-size: 13px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: rgba(16, 185, 129, 0.1);
                border-radius: 4px;
            """)
            self.conn_status_indicator.setStyleSheet("""
                QLabel {
                    color: #10b981;
                    font-size: 20px;
                    background: transparent;
                }
            """)
            self.conn_status_indicator.setGraphicsEffect(self._create_glow_effect("#10b981"))

            self.pause_btn.setEnabled(True)
            self.connect_btn.setText('🔌 断开')
            self.debug_label.setText('数据接收中...')
            log_emitter.log(f"串口连接成功: {message}", "success")

            # ============ 记录连接事件到MAT会话 ============
            if self.mat_session_active:
                self.session_manager.append_event('eeg_connected', 'CONNECT', f'EEG device connected: {message}')
        else:
            # 更新连接状态指示器（红色）
            self.conn_status_label.setText("未连接")
            self.conn_status_label.setStyleSheet("""
                color: #ef4444;
                font-size: 13px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: rgba(239, 68, 68, 0.1);
                border-radius: 4px;
            """)
            self.conn_status_indicator.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    font-size: 20px;
                    background: transparent;
                }
            """)
            self.conn_status_indicator.setGraphicsEffect(self._create_glow_effect("#ef4444"))
            self.connect_btn.setText('🔌 打开')
            self.pause_btn.setEnabled(False)
            log_emitter.log(f"串口断开: {message}", "warning")

            # ============ 记录断开事件到MAT会话 ============
            if self.mat_session_active:
                self.session_manager.append_event('eeg_disconnected', 'DISCONNECT', f'EEG device disconnected: {message}')

    def _split_display_and_inference_values(self, corrected_values):
        """将输入数据拆成模型推理用完整通道和界面显示用8通道。"""
        values = [float(v) for v in corrected_values]
        n = len(values)
        if n >= 62:
            inference_values = values[:62]
            display_values = [values[i] for i in DISPLAY_CHANNEL_INDICES_62]
        elif n == 8:
            inference_values = values[:8]
            display_values = values[:8]
        else:
            if n > 8:
                inference_values = values
                display_values = values[:8]
                if self.packet_count % 1000 == 0:
                    log_emitter.log(f"检测到 {n} 通道输入，界面仅显示前8通道。", "warning")
            else:
                inference_values = values
                display_values = values + [0.0] * (8 - n)
                if self.packet_count % 1000 == 0:
                    log_emitter.log(f"检测到 {n} 通道输入，不足8通道，显示端已补零。", "warning")
        return inference_values, display_values

    def _reset_full_buffers(self, input_channels):
        self.input_num_channels = input_channels
        self.full_data_buffers = [np.zeros(1000) for _ in range(input_channels)]
        self.full_filtered_buffers = [np.zeros(1000) for _ in range(input_channels)]

    def _current_model_type(self):
        return getattr(self.model_thread, "model_type", "") if hasattr(self, "model_thread") else ""

    def update_data(self, values):
        """更新显示数据（包括时域和频域）"""
        if self.is_paused:
            return

        self.packet_count += 1
        if self.is_playback_mode and self.prediction_mode_enabled:
            current_time_sec = max(self.packet_count - 1, 0) / SAMPLE_RATE
            self._sync_prediction_display(current_time_sec)

        # 应用数据缩放修正
        # 离线回放模式：CSV数据已经是修正后的值，不需要再次修正
        # 实时采集模式：正常模式除以24，0Ω修正模式除以48
        if self.is_playback_mode:
            corrected_values = [float(v) for v in values]
            scale_factor = 1.0
        else:
            scale_factor = 48.0 if self.zero_ohm_correction else 24.0
            corrected_values = [float(v) / scale_factor for v in values]

        inference_values, display_values = self._split_display_and_inference_values(corrected_values)
        input_channels = len(inference_values)
        display_channels = len(display_values)

        # 首次接收数据，自动检测输入通道数量；62通道输入只创建8条显示曲线。
        if self.num_channels == 0 or self.input_num_channels != input_channels:
            if self.num_channels != display_channels:
                self.create_channels(display_channels)
            self._reset_full_buffers(input_channels)
            if input_channels >= 62:
                log_emitter.log("已加载 62 通道离线回放数据，界面显示 8 通道，模型推理使用完整 62 通道。", "success")
            else:
                log_emitter.log(f"检测到 {input_channels} 通道数据，界面显示 {display_channels} 通道。", "info")

        if self._current_model_type() in ("uda-dda-online", "uda-dda-binary") and input_channels != 62 and self.packet_count % 1000 == 1:
            log_emitter.log(
                f"当前 UDA-DDA 在线模型期望 62 通道输入，当前输入为 {input_channels} 通道，可能无法正常推理。",
                "warning"
            )

        # 写入CSV文件（如果正在记录）
        if self.is_recording and self.csv_writer:
            try:
                # 计算时间戳（相对于记录开始时间，单位：秒）
                if self.recording_start_time:
                    elapsed_time = (datetime.now() - self.recording_start_time).total_seconds()
                    elapsed_time_str = f"{elapsed_time:.3f}"
                else:
                    elapsed_time_str = "0.000"

                # 写入一行数据：时间戳 + 各通道的值
                row = [elapsed_time_str] + [f"{v:.2f}" for v in corrected_values]
                self.csv_writer.writerow(row)

                # 每100行flush一次，确保数据写入磁盘
                if self.packet_count % 100 == 0:
                    self.csv_file.flush()

            except Exception as e:
                log_emitter.log(f"写入CSV失败: {e}", "error")
                self.stop_recording()

        # 每100包更新一次标签显示
        if self.packet_count % 100 == 0:
            # 更新数据包计数
            self.packet_count_label.setText(f'数据包: {self.packet_count}')

            # 使用映射关系生成通道信息
            ch_info_parts = []
            for i in range(min(4, len(display_values))):
                if i < len(self.physical_channels):
                    physical_ch = self.physical_channels[i]
                    electrode_pos = self.channel_mapping.get(physical_ch, '未使用')
                    if electrode_pos != '未使用':
                        ch_name = f'{electrode_pos} ({physical_ch})'
                    else:
                        ch_name = physical_ch
                    ch_info_parts.append(f'{ch_name}: {display_values[i]:.1f}µV')
                else:
                    ch_info_parts.append(f'CH{i+1}: {display_values[i]:.1f}µV')

            ch_info = ' | '.join(ch_info_parts)
            if len(display_values) > 4:
                ch_info += f' ... (+{len(display_values)-4}ch)'

            # 根据模式显示不同的信息
            if self.is_playback_mode:
                mode = "离线回放"
                playback_progress = f' | 进度: {self.playback_progress.value()}%'
            else:
                mode = "信号发生器测试" if self.zero_ohm_correction else "正常脑电"
                playback_progress = ""

            record_status = " [REC]" if self.is_recording else ""
            self.debug_label.setText(f'{ch_info} | [{mode}]{playback_progress}{record_status}')

        # 更新数据缓冲区 - 实现滚动效果
        # 添加安全检查：确保 data_buffers 已正确初始化
        if self.num_channels > 0 and len(self.data_buffers) >= self.num_channels:
            for i in range(self.num_channels):
                if i < len(display_values):
                    # 左移数据，添加新数据到末尾
                    self.data_buffers[i][:-1] = self.data_buffers[i][1:]
                    self.data_buffers[i][-1] = display_values[i]

            if len(self.full_data_buffers) >= input_channels:
                for i in range(input_channels):
                    self.full_data_buffers[i][:-1] = self.full_data_buffers[i][1:]
                    self.full_data_buffers[i][-1] = inference_values[i]

            # ============ 追加EEG数据到MAT会话 ============
            if self.mat_session_active:
                import time
                samples = [display_values[i] for i in range(min(len(display_values), self.num_channels))]
                timestamp = time.time()
                self.session_manager.append_eeg(samples, timestamp)

        # ============ EEG 实时滤波流水线（每5包滤波一次，降低CPU占用）============
        if self.packet_count % 5 == 0 and self.num_channels > 0 and len(self.filtered_buffers) >= self.num_channels:
            for i in range(self.num_channels):
                # 提取原始数据缓冲区
                raw_buffer = self.data_buffers[i]

                # 50Hz 陷波滤波（滤除市电干扰）
                notch_filtered = filtfilt(self.b_notch, self.a_notch, raw_buffer)

                # 0.5-50Hz 带通滤波（滤除基线漂移和高频噪声）
                band_filtered = filtfilt(self.b_band, self.a_band, notch_filtered)

                # 更新滤波后的数据缓冲区
                self.filtered_buffers[i][:] = band_filtered

                # 更新时域曲线（使用滤波后的数据）
                self.curves[i].setData(self.filtered_buffers[i])

            if len(self.full_data_buffers) >= self.input_num_channels and len(self.full_filtered_buffers) >= self.input_num_channels:
                for i in range(self.input_num_channels):
                    raw_buffer = self.full_data_buffers[i]
                    notch_filtered = filtfilt(self.b_notch, self.a_notch, raw_buffer)
                    band_filtered = filtfilt(self.b_band, self.a_band, notch_filtered)
                    self.full_filtered_buffers[i][:] = band_filtered

        # ============ 发起异步决策引擎推理（每100包，0.5秒步长）============
        # 离线训练步长：0.5秒 = 100个采样点@200Hz
        if (not self.prediction_mode_enabled) and (self.model_selector_combo.currentData() != PREDICTION_SYNC_SOURCE_TYPE) and self.packet_count % 100 == 0 and self.input_num_channels > 0 and len(self.full_filtered_buffers) >= self.input_num_channels:
            # 提取EEG数据矩阵：(时间点, 通道数)
            # 使用滤波后缓冲区的最后200个点（1秒数据@200Hz）
            eeg_matrix = np.array([self.full_filtered_buffers[i][-200:] for i in range(self.input_num_channels)]).T

            now = time.time()
            if self.model_thread.is_busy:
                if now - self.last_inference_skip_log_time >= self.inference_submit_interval_sec:
                    log_emitter.log("模型仍在推理，跳过当前窗口提交。", "model")
                    self.last_inference_skip_log_time = now
            elif now - self.last_inference_submit_time >= self.inference_submit_interval_sec:
                self.model_thread.do_inference(eeg_matrix)
                self.last_inference_submit_time = now
            elif now - self.last_inference_skip_log_time >= self.inference_submit_interval_sec:
                log_emitter.log(
                    f"跳过推理提交：距离上次提交不足 {self.inference_submit_interval_sec:.0f} 秒",
                    "model"
                )
                self.last_inference_skip_log_time = now

        # ============ 更新频带能量柱状图和DE矩阵 - 每10包更新一次 ============
        if self.packet_count % 10 == 0:
            try:
                from scipy.signal import welch

                sample_rate = 200  # Hz - 与离线训练对齐
                epsilon = 1e-10  # 防止log(0)

                # 定义频带范围 (Hz) - 严格定义，Delta从0.5Hz开始，避免包含直流分量
                band_ranges = [
                    (0.5, 4),    # Delta: 0.5-4Hz
                    (4, 8),      # Theta: 4-8Hz
                    (8, 13),     # Alpha: 8-13Hz
                    (13, 30),    # Beta: 13-30Hz
                    (30, 50)     # Gamma: 30-50Hz
                ]

                # 存储所有通道的频带能量（用于计算平均值和DE）
                all_channel_band_powers = []  # shape: (num_channels, 5)

                # 提取最新1秒数据（200个点）
                window_size = 200

                # 添加安全检查：确保 filtered_buffers 已正确初始化
                if self.num_channels > 0 and len(self.filtered_buffers) >= self.num_channels:
                    for ch in range(self.num_channels):
                        # 提取最新1秒的滤波后数据
                        data = self.filtered_buffers[ch][-window_size:]

                        # 强制去直流分量
                        data = data - np.mean(data)

                        # ============ 使用 Welch 方法计算 PSD ============
                        # nperseg=200 (1秒窗口), noverlap=None (默认50%重叠)
                        freqs, psd = welch(
                            data,
                            fs=sample_rate,
                            nperseg=200,
                            noverlap=None,
                            scaling='density'  # 返回功率谱密度
                        )

                        # 计算每个频带的能量（PSD积分）
                        band_powers = []
                        for low, high in band_ranges:
                            # 严格定义频带掩码（不包含边界，避免重叠）
                            band_mask = (freqs >= low) & (freqs < high)
                            # 使用梯形积分计算频带总功率
                            band_power = np.trapz(psd[band_mask], freqs[band_mask])
                            band_powers.append(band_power)

                        all_channel_band_powers.append(band_powers)

                # 转换为 numpy 数组方便计算
                all_channel_band_powers = np.array(all_channel_band_powers)  # (num_channels, 5)

                # ============ 更新8通道独立频谱图 ============
                from scipy.ndimage import gaussian_filter1d

                # 添加安全检查：确保 filtered_buffers 已正确初始化
                if self.num_channels > 0 and len(self.filtered_buffers) >= self.num_channels:
                    # 遍历 8 个通道，分别计算 PSD
                    for ch in range(min(8, self.num_channels)):
                        # 提取该通道最新 1 秒数据
                        data = self.filtered_buffers[ch][-window_size:] - np.mean(self.filtered_buffers[ch][-window_size:])

                        # 使用 Welch 方法计算 PSD
                        freqs, psd = welch(
                            data,
                            fs=sample_rate,
                            nperseg=200,
                            noverlap=None,
                            scaling='density'
                        )

                        # 只取 0.5-50Hz 范围内的频率
                        freq_mask = (freqs >= 0.5) & (freqs <= 50)
                        display_freqs = freqs[freq_mask]
                        display_psd = psd[freq_mask]

                        # 转换为 dB 尺度
                        log_psd = 10 * np.log10(display_psd + epsilon)

                        # 使用高斯滤波平滑曲线（sigma=1.0）
                        smoothed_log_psd = gaussian_filter1d(log_psd, sigma=1.0)

                    # 更新该通道的频谱曲线
                    if ch < len(self.fft_curves):
                        self.fft_curves[ch].setData(display_freqs, smoothed_log_psd)

            except Exception as e:
                pass  # 静默处理错误

        # Y轴量程控制 - 根据下拉框选择设置范围（使用滤波后的数据）
        if self.packet_count % 10 == 0:  # 每10包更新一次
            for i in range(self.num_channels):
                if self.y_axis_range == 'auto':
                    # 自动缩放模式：根据滤波后数据范围动态调整
                    data_min = np.min(self.filtered_buffers[i])
                    data_max = np.max(self.filtered_buffers[i])
                    margin = (data_max - data_min) * 0.15  # 15%边距
                    if data_max - data_min > 0.001:  # 避免除零
                        self.plots[i].setYRange(data_min - margin, data_max + margin, padding=0)
                else:
                    # 固定量程模式：使用下拉框选择的固定范围
                    limit = self.y_range_limits[self.y_axis_range]
                    self.plots[i].setYRange(-limit, limit, padding=0)

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        log_emitter.log("程序正在关闭...", "warning")

        # 如果正在记录，先停止记录并关闭文件
        if self.is_recording:
            log_emitter.log("停止数据记录...", "info")
            self.stop_recording()

        # 停止决策引擎推理线程
        if self.model_thread and self.model_thread.is_alive():
            self.model_thread.stop()
            self.model_thread.join(timeout=2.0)

        # 关闭UDP
        if self.udp_sender:
            self.udp_sender.close()

        # 关闭串口
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.stop()
            self.serial_thread.join(timeout=1.0)

        # 停止回放线程
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.stop()
            self.playback_thread.join(timeout=1.0)

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = EEGDisplayWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
