#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCI 闭环神经调控系统 - 模型适配器层
=============================================

功能：
1. 定义统一的模型接口 BaseModelAdapter
2. 实现 SVMAdapter（传统 DE 特征 + SVM）
3. 实现 DeepLearningAdapter（PyTorch 深度学习模型模板）

设计模式：适配器模式（Adapter Pattern）
- 将不同类型的模型（SVM、深度学习）统一为相同接口
- 实现模型的"即插即用"

作者：BCI Lab
日期：2025
"""

import numpy as np
import pickle
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

try:
    from online_emotion_engine import OnlineEmotionEngine, DEFAULT_UDA_DDA_MODEL_PATH, DEFAULT_SCALER_PATH
except Exception:
    OnlineEmotionEngine = None
    DEFAULT_UDA_DDA_MODEL_PATH = os.environ.get(
        "UDA_DDA_MODEL_PATH",
        os.path.join("models", "subject15_calib_supervised_best_model.pth"),
    )
    DEFAULT_SCALER_PATH = os.environ.get(
        "UDA_DDA_SCALER_PATH",
        os.path.join("models", "calibration_raw_scaler.pkl"),
    )

# 尝试导入 sklearn 和 joblib（如果失败会打印警告）
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: 未安装 sklearn 或 joblib，SVM功能将不可用")


class BaseModelAdapter(ABC):
    """
    模型适配器基类

    定义统一的模型接口，所有模型适配器必须继承此类并实现 predict 方法。
    """

    def __init__(self, model_path: str, log_callback=None):
        """
        初始化适配器

        Args:
            model_path: 模型文件路径
            log_callback: 日志回调函数
        """
        self.model_path = model_path
        self.log_callback = log_callback
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def predict(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> Tuple[float, str, int, np.ndarray]:
        """
        统一的预测接口

        Args:
            eeg_matrix: 脑电数据矩阵，shape = (n_samples, n_channels)
                       例如：(200, 8) 表示 1 秒数据，8 个通道 @ 200Hz
            fs: 采样率，默认 200 Hz

        Returns:
            tuple: (score, ui_emotion, class_idx, probabilities)
                - score: float, 负向情绪得分 [0-100]
                - ui_emotion: str, UI 显示的情绪标签
                - class_idx: int, 预测类别索引（0:负向, 1:中性, 2:正向）
                - probabilities: np.ndarray, 三分类概率数组 [prob_neg, prob_neu, prob_pos]
        """
        pass

    def _log(self, message: str):
        """日志输出"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)


class SVMAdapter(BaseModelAdapter):
    """
    SVM 模型适配器

    功能：
    1. 接收 8 通道脑电矩阵，提取 40 维微分熵（DE）特征
    2. 实现流式滑动平均平滑 (Streaming Moving Average，消除瞬态噪声)
    3. 支持在线被试独立标准化 (Online Subject-wise Z-score，消除个体基线差异)
    4. 调用 SVM 模型进行三分类预测
    """

    def __init__(
            self,
            model_path: str = os.path.join("models", "svm_de_8ch_baseline_model.pkl"),
            scaler_path: Optional[str] = None,
            log_callback=None,
            smoothing_window: int = 10  # 对应离线训练时的 10 个滑动窗口
    ):
        super().__init__(model_path, log_callback)

        # 严格对应训练代码的 5 个频带
        self.band_ranges = [
            (0.5, 4),  # Delta
            (4, 8),  # Theta
            (8, 13),  # Alpha
            (13, 30),  # Beta
            (30, 50)  # Gamma
        ]

        # --- 核心对齐 1：流式平滑缓冲区 ---
        self.smoothing_window = smoothing_window
        self.feature_buffer = deque(maxlen=self.smoothing_window)

        # --- 核心对齐 2：在线标定参数 ---
        self.is_calibrating = False
        self.calibration_features = []  # 用于收集 60 秒标定态的平滑特征
        self.online_scaler = None  # 为当前佩戴者动态生成的标准化器

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载 SVM 模型"""
        if not SKLEARN_AVAILABLE:
            self._log("[SVMAdapter] ❌ sklearn 或 joblib 未安装，无法加载模型")
            self.is_loaded = False
            return

        try:
            if os.path.exists(self.model_path):
                # 使用 joblib 加载（兼容训练代码的保存方式）
                model_dict = joblib.load(self.model_path)

                if isinstance(model_dict, dict) and 'model' in model_dict:
                    self.model = model_dict['model']
                    # 注：离线保存的 dummy_scaler 这里不需要用，因为我们要实现在线 scaler
                    self._log(f"[SVMAdapter] 成功加载模型: {self.model_path}")
                else:
                    self.model = model_dict
                    self._log(f"[SVMAdapter] 成功加载模型（旧格式）: {self.model_path}")
                self.is_loaded = True
            else:
                self._log(f"[SVMAdapter] ❌ 模型文件不存在: {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            self._log(f"[SVMAdapter] ❌ 加载模型失败: {e}")
            self.is_loaded = False

    def extract_differential_entropy_features(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> np.ndarray:
        """提取基础微分熵（DE）特征 (严格对应离线提取逻辑)"""
        epsilon = 1e-8
        n_channels = eeg_matrix.shape[1]  # 此时送进来的已经是干净的 8 通道
        de_features = []

        for ch in range(n_channels):
            channel_data = eeg_matrix[:, ch]
            # 去直流
            channel_data = channel_data - np.mean(channel_data)
            ch_band_features = []

            for low, high in self.band_ranges:
                sos = butter(4, [low, high], btype='bandpass', output='sos', fs=fs)
                filtered_data = sosfiltfilt(sos, channel_data)
                var = np.var(filtered_data)
                de = 0.5 * np.log(2 * np.pi * np.e * var + epsilon)
                ch_band_features.append(de)

            de_features.extend(ch_band_features)

        return np.array(de_features)  # shape: (40,)

    # ==========================================
    # 以下为配合决策引擎 FSM 的在线标定接口
    # ==========================================
    def start_calibration(self):
        """进入标定模式：重置并开始收集当前用户的基线特征"""
        self.is_calibrating = True
        self.calibration_features = []
        self._log("[SVMAdapter] 🔄 进入在线标定模式，开始收集个体基线特征...")

    def finish_calibration(self):
        """结束标定：拟合该用户专属的在线标准化器"""
        self.is_calibrating = False

        if not SKLEARN_AVAILABLE:
            self._log("[SVMAdapter] ⚠️ 警告：sklearn 未安装，跳过在线标定")
            return

        if len(self.calibration_features) > 0:
            calib_data = np.vstack(self.calibration_features)  # shape: (N, 40)
            self.online_scaler = StandardScaler()
            self.online_scaler.fit(calib_data)
            self._log(
                f"[SVMAdapter] ✅ 标定完成，成功拟合个体专属 Z-score 模型 (样本数: {len(self.calibration_features)})")
        else:
            self._log("[SVMAdapter] ⚠️ 警告：未收集到标定数据！")

    def predict(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> Tuple[float, str, int, np.ndarray]:
        """
        统一预测接口

        Returns:
            tuple: (score, ui_emotion, class_idx, probabilities)
                - score: float, 负向情绪得分 [0-100]
                - ui_emotion: str, UI 显示的情绪标签
                - class_idx: int, 预测类别索引（0:负向, 1:中性, 2:正向）
                - probabilities: np.ndarray, 三分类概率数组 [prob_neg, prob_neu, prob_pos]
        """
        # 默认返回值（错误或未加载时）
        default_probs = np.array([0.333, 0.333, 0.334])

        if not self.is_loaded or self.model is None:
            return 50.0, "Unknown", 1, default_probs

        try:
            # 1. 提取单帧 40 维基础特征
            raw_features = self.extract_differential_entropy_features(eeg_matrix, fs)

            # 2. 流式滑动平均平滑 (对齐训练时的 moving_average_smoothing)
            self.feature_buffer.append(raw_features)
            smoothed_features = np.mean(list(self.feature_buffer), axis=0).reshape(1, -1)

            # 3. 状态路由
            if self.is_calibrating:
                # 标定状态：收集平滑后的特征，不输出真实情绪（返回占位符）
                self.calibration_features.append(smoothed_features[0])
                return 50.0, "Calibrating", 1, default_probs

            else:
                # 监测状态：执行在线标准化并预测
                features_to_predict = smoothed_features

                # 使用刚刚拟合好的个人专属 Scaler
                if self.online_scaler is not None:
                    features_to_predict = self.online_scaler.transform(smoothed_features)

                # 4. SVM 预测概率 (提取 class 0 负向情绪的概率作为最终得分)
                probs = self.model.predict_proba(features_to_predict)[0]
                score = probs[0] * 100
                class_idx = int(np.argmax(probs))

                emotion_map = {0: "Negative (负向)", 1: "Neutral (中性)", 2: "Positive (正向)"}
                ui_emotion = emotion_map.get(class_idx, "Unknown")

                return float(score), ui_emotion, class_idx, probs

        except Exception as e:
            self._log(f"[SVMAdapter] ❌ 推理流异常: {e}")
            return 50.0, "Error", 1, default_probs

class DeepLearningAdapter(BaseModelAdapter):
    """
    深度学习模型适配器（模板/占位符）

    功能：
    1. 接收脑电矩阵，转换为 PyTorch Tensor
    2. 调用深度学习模型进行推理
    3. 返回统一格式的预测结果

    支持的模型：
    - UDA-DDA
    - EEGNet
    - SDC-Net
    - 其他基于 PyTorch 的 EEG 分类模型
    """

    def __init__(
        self,
        model_path: str,
        model_class=None,
        log_callback=None,
        device: str = 'cpu'
    ):
        """
        初始化深度学习适配器

        Args:
            model_path: 模型权重文件路径（.pth, .pt, .pkl 等）
            model_class: 模型类（需要预先定义，例如 EEGNet, UDA-DDA 等）
            log_callback: 日志回调函数
            device: 运行设备，'cpu' 或 'cuda'
        """
        super().__init__(model_path, log_callback)
        self.model_class = model_class
        self.device = device
        self.model = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载深度学习模型"""
        try:
            import torch

            if self.model_class is None:
                self._log("[DeepLearningAdapter] 警告: 未指定 model_class，无法加载模型")
                self.is_loaded = False
                return

            if os.path.exists(self.model_path):
                # 创建模型实例
                self.model = self.model_class()

                # 加载权重
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()

                self._log(f"[DeepLearningAdapter] 成功加载模型: {self.model_path}")
                self.is_loaded = True
            else:
                self._log(f"[DeepLearningAdapter] 模型文件不存在: {self.model_path}")
                self.is_loaded = False

        except ImportError:
            self._log("[DeepLearningAdapter] 未安装 PyTorch，无法使用深度学习模型")
            self.is_loaded = False
        except Exception as e:
            self._log(f"[DeepLearningAdapter] 加载模型失败: {e}")
            self.is_loaded = False

    def preprocess(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> 'torch.Tensor':
        """
        数据预处理：将 numpy 数组转换为 PyTorch Tensor

        Args:
            eeg_matrix: 脑电数据矩阵，shape = (n_samples, n_channels)
            fs: 采样率

        Returns:
            torch.Tensor: 模型输入张量
                           例如：(1, 1, 8, 200) 表示 batch_size=1, channels=1, EEG_channels=8, time_points=200
        """
        try:
            import torch

            # 转换为 float32
            data = eeg_matrix.astype(np.float32)

            # 转换为 Tensor
            data_tensor = torch.from_numpy(data)

            # 调整维度：(n_samples, n_channels) -> (1, 1, n_channels, n_samples)
            # 这是因为大多数 EEG 深度学习模型期望输入格式为 (batch, channel, EEG_channel, time)
            data_tensor = data_tensor.permute(1, 0).unsqueeze(0).unsqueeze(0)

            # 移动到指定设备
            data_tensor = data_tensor.to(self.device)

            return data_tensor

        except Exception as e:
            self._log(f"[DeepLearningAdapter] 预处理失败: {e}")
            return None

    def predict(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> Tuple[float, str, int]:
        """
        使用深度学习模型进行预测

        Args:
            eeg_matrix: 脑电数据矩阵，shape = (n_samples, n_channels)
            fs: 采样率，默认 200 Hz

        Returns:
            tuple: (score, ui_emotion, class_idx)
        """
        if not self.is_loaded or self.model is None:
            # 模型未加载，返回虚拟值用于测试
            score = np.clip(np.random.normal(50, 15), 0, 100)
            class_idx = np.random.randint(0, 3)
            emotion_map = ["Negative (负向)", "Neutral (中性)", "Positive (正向)"]
            return float(score), emotion_map[class_idx], class_idx

        try:
            import torch

            # 预处理数据
            input_tensor = self.preprocess(eeg_matrix, fs)
            if input_tensor is None:
                return 50.0, "Unknown", 1

            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)

                # 获取预测概率（假设模型输出 logits 或概率）
                if outputs.shape[-1] == 3:
                    # 三分类
                    probs = torch.softmax(outputs, dim=-1)[0].cpu().numpy()
                else:
                    # 二分类或其他情况
                    probs = torch.sigmoid(outputs).cpu().numpy()

            # 提取负向情绪概率作为得分
            score = probs[0] * 100 if len(probs) > 0 else 50.0

            # 获取预测类别
            class_idx = int(np.argmax(probs))

            # 生成 UI 情绪标签
            emotion_map = {
                0: "Negative (负向)",
                1: "Neutral (中性)",
                2: "Positive (正向)"
            }
            ui_emotion = emotion_map.get(class_idx, "Unknown")

            return float(score), ui_emotion, class_idx

        except Exception as e:
            self._log(f"[DeepLearningAdapter] 预测失败: {e}，返回默认值")
            return 50.0, "Unknown", 1


# ============ 模型适配器工厂 ============
class UDADDAOnlineAdapter(BaseModelAdapter):
    """
    UDA-DDA online adapter for eeg_viewer2.py.

    It wraps OnlineEmotionEngine and keeps the GUI-facing call signature:
        score, ui_emotion, class_idx, probs = adapter.predict(eeg_matrix, fs=200)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_UDA_DDA_MODEL_PATH,
        log_callback=None,
        output_mode: str = "three_class",
        input_dim: int = 310,
        **engine_kwargs
    ):
        super().__init__(model_path, log_callback)
        if OnlineEmotionEngine is None:
            raise ImportError("Cannot import OnlineEmotionEngine from online_emotion_engine.py")

        self.output_mode = output_mode
        self.engine = OnlineEmotionEngine(
            model_path=model_path,
            input_dim=input_dim,
            output_mode=output_mode,
            log_callback=log_callback,
            **engine_kwargs
        )
        self.is_loaded = True
        self._log(f"Adapter created: {self.__class__.__name__}")
        self._log(f"model_path: {model_path}")
        self._log(f"scaler_path: {engine_kwargs.get('scaler_path')}")
        self._log(f"using_identity_scaler: {getattr(self.engine, 'using_identity_scaler', None)}")

    def predict(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> Tuple[float, str, int, np.ndarray]:
        result = self.engine.predict(eeg_matrix, fs=fs)
        return (
            float(result["score"]),
            result["ui_emotion"],
            int(result["class_idx"]),
            np.asarray(result["probabilities"], dtype=np.float32),
        )


class AdapterFactory:
    """
    模型适配器工厂

    根据模型类型自动创建对应的适配器实例
    """

    # 默认模型路径配置
    DEFAULT_MODEL_PATHS = {
        'svm': os.path.join("models", "svm_de_8ch_baseline_model.pkl"),
        'eegnet': os.path.join("models", "eegnet_8ch.pth"),
        'uda-dda-online': DEFAULT_UDA_DDA_MODEL_PATH,
        'uda-dda-binary': DEFAULT_UDA_DDA_MODEL_PATH,
    }

    @staticmethod
    def create_adapter(
        model_type: str,
        model_path: str = None,
        **kwargs
    ) -> BaseModelAdapter:
        """
        创建模型适配器

        Args:
            model_type: 模型类型
                        - 'svm': 传统 SVM 模型（8ch 频域+Z-score）
                        - 'eegnet': EEGNet 深度学习模型（8ch 时域波形）
            model_path: 模型文件路径（可选，默认使用预设路径）
            **kwargs: 其他参数（log_callback, model_class, device 等）

        Returns:
            BaseModelAdapter: 模型适配器实例
        """
        model_type = model_type.lower()
        if model_type == 'uda-dda':
            model_type = 'uda-dda-online'

        # 如果未指定模型路径，使用默认路径
        if model_path is None:
            model_path = AdapterFactory.DEFAULT_MODEL_PATHS.get(model_type)
            if model_path is None:
                raise ValueError(f"未指定模型路径，且模型类型 '{model_type}' 没有默认路径")

        if model_type == 'svm':
            return SVMAdapter(
                model_path=model_path,
                scaler_path=kwargs.get('scaler_path'),
                log_callback=kwargs.get('log_callback')
            )

        elif model_type == 'uda-dda-online':
            scaler_path = kwargs.get('scaler_path') or DEFAULT_SCALER_PATH
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler file not found: {scaler_path}. "
                    "Please run prepare_upper_demo_scaler.py first."
                )
            return UDADDAOnlineAdapter(
                model_path=model_path,
                log_callback=kwargs.get('log_callback'),
                output_mode='three_class',
                input_dim=kwargs.get('input_dim', 310),
                scaler_path=scaler_path,
                calibration_features=kwargs.get('calibration_features'),
                scaler_postprocess=kwargs.get('scaler_postprocess', 'clip'),
                prob_smoother_type=kwargs.get('prob_smoother_type', 'ema'),
                prob_ema_alpha=kwargs.get('prob_ema_alpha', 0.05),
                prob_ma_window=kwargs.get('prob_ma_window', 5),
                prob_median_window=kwargs.get('prob_median_window', 5),
                device=kwargs.get('device', None),
            )

        elif model_type == 'uda-dda-binary':
            scaler_path = kwargs.get('scaler_path') or DEFAULT_SCALER_PATH
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler file not found: {scaler_path}. "
                    "Please run prepare_upper_demo_scaler.py first."
                )
            return UDADDAOnlineAdapter(
                model_path=model_path,
                log_callback=kwargs.get('log_callback'),
                output_mode='binary_negative',
                input_dim=kwargs.get('input_dim', 310),
                scaler_path=scaler_path,
                calibration_features=kwargs.get('calibration_features'),
                scaler_postprocess=kwargs.get('scaler_postprocess', 'clip'),
                negative_threshold=kwargs.get('negative_threshold', 0.5),
                prob_smoother_type=kwargs.get('prob_smoother_type', 'ema'),
                prob_ema_alpha=kwargs.get('prob_ema_alpha', 0.05),
                prob_ma_window=kwargs.get('prob_ma_window', 5),
                prob_median_window=kwargs.get('prob_median_window', 5),
                device=kwargs.get('device', None),
            )

        elif model_type == 'eegnet':
            # EEGNet 使用 EEGNetPro_Adapter 作为模型类
            model_class = kwargs.get('model_class', EEGNetPro_Adapter)
            return DeepLearningAdapter(
                model_path=model_path,
                model_class=model_class,
                log_callback=kwargs.get('log_callback'),
                device=kwargs.get('device', 'cpu')
            )

        elif model_type in ['uda-dda', 'sdc-net', 'deep']:
            # 其他深度学习模型
            return DeepLearningAdapter(
                model_path=model_path,
                model_class=kwargs.get('model_class'),
                log_callback=kwargs.get('log_callback'),
                device=kwargs.get('device', 'cpu')
            )

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class EEGNetPro_Adapter(nn.Module):
    def __init__(self, num_classes=3, num_channels=8, F1=16, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F2, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F2, num_classes)
        )

    def forward(self, x):
        # 此时输入的 x 形状应为 (Batch, 1, Channels, Time)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.classifier(x)
        return x


# --- 2. 编写 EEGNet 的适配器逻辑 ---
class DeepLearningAdapter(BaseModelAdapter):
    """
    EEGNet 深度学习模型适配器
    功能：接收 8 通道 1 秒原始脑电，转为 PyTorch 张量进行实时推理
    """

    def __init__(self, model_path: str = os.path.join("models", "eegnet_8ch.pth"), log_callback=None):
        super().__init__(model_path, log_callback)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 深度学习通常在模型内部处理时序关系，但为了系统兼容性，我们保留标定接口
        self.is_calibrating = False
        self.calibration_features = []
        self.online_scaler = None

        self._load_model()

    def _load_model(self):
        try:
            self.model = EEGNetPro_Adapter(num_classes=3, num_channels=8).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()  # 切换到推理模式
            self.is_loaded = True
            self._log(f"[EEGNetAdapter] ✅ 成功加载深度学习模型: {self.model_path} (运行在 {self.device})")
        except Exception as e:
            self._log(f"[EEGNetAdapter] ❌ 加载模型失败: {e}")
            self.is_loaded = False

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_features = []
        self._log("[EEGNetAdapter] 🔄 开始收集基线波形...")

    def finish_calibration(self):
        self.is_calibrating = False
        if len(self.calibration_features) > 0:
            calib_data = np.vstack(self.calibration_features)  # shape: (N, 200*8)
            self.online_scaler = StandardScaler()
            self.online_scaler.fit(calib_data)
            self._log("[EEGNetAdapter] ✅ 标定完成，已建立时域波形 Z-score 模型")

    def predict(self, eeg_matrix: np.ndarray, fs: float = 200.0) -> Tuple[float, str, int, np.ndarray]:
        """
        统一预测接口

        Returns:
            tuple: (score, ui_emotion, class_idx, probabilities)
                - score: float, 负向情绪得分 [0-100]
                - ui_emotion: str, UI 显示的情绪标签
                - class_idx: int, 预测类别索引（0:负向, 1:中性, 2:正向）
                - probabilities: np.ndarray, 三分类概率数组 [prob_neg, prob_neu, prob_pos]
        """
        # 默认返回值（错误或未加载时）
        default_probs = np.array([0.333, 0.333, 0.334])

        if not self.is_loaded or self.model is None:
            return 50.0, "Unknown", 1, default_probs

        try:
            # 1. 在线标定路由 (针对原始波形)
            flat_eeg = eeg_matrix.flatten().reshape(1, -1)

            if self.is_calibrating:
                self.calibration_features.append(flat_eeg[0])
                return 50.0, "Calibrating", 1, default_probs

            # 2. 波形标准化 (如果已经完成标定)
            if self.online_scaler is not None:
                norm_flat = self.online_scaler.transform(flat_eeg)
                norm_matrix = norm_flat.reshape(200, 8)
            else:
                norm_matrix = eeg_matrix

            # 3. 转换为 PyTorch 张量
            # 模型需要形状：(Batch=1, 1, Channels=8, Time=200)
            # 当前 norm_matrix 形状：(200, 8)
            tensor_data = torch.tensor(norm_matrix.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

            # 4. 模型推理
            with torch.no_grad():
                outputs = self.model(tensor_data)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            # 提取得分和情绪
            score = probs[0] * 100  # 负向情绪概率 * 100
            class_idx = int(np.argmax(probs))

            emotion_map = {0: "Negative (负向)", 1: "Neutral (中性)", 2: "Positive (正向)"}
            return float(score), emotion_map.get(class_idx, "Unknown"), class_idx, probs

        except Exception as e:
            self._log(f"[EEGNetAdapter] ❌ 推理流异常: {e}")
            return 50.0, "Error", 1, default_probs

# ============ 测试代码 ============
if __name__ == '__main__':

    """
    测试块：验证适配器接口一致性
    """
    import time

    print("\n" + "=" * 60)
    print("模型适配器测试程序")
    print("=" * 60 + "\n")

    # 生成测试数据（1 秒 EEG 数据，8 通道，200 Hz）
    np.random.seed(42)
    test_eeg = np.random.randn(200, 8)

    print(f"测试数据形状: {test_eeg.shape}")
    print(f"采样率: 200 Hz")
    print(f"数据时长: 1 秒\n")

    # 测试 SVM 适配器
    print("-" * 60)
    print("测试 SVM 适配器")
    print("-" * 60)

    svm_adapter = SVMAdapter(
        model_path="seed_de_8ch_baseline_model.pkl",
        log_callback=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    )

    score, emotion, class_idx, probs = svm_adapter.predict(test_eeg, fs=200.0)
    print(f"预测结果: score={score:.2f}, emotion={emotion}, class_idx={class_idx}")
    print(f"概率分布: 负向={probs[0]*100:.1f}%, 中性={probs[1]*100:.1f}%, 正向={probs[2]*100:.1f}%")

    # 测试深度学习适配器（虚拟模型）
    print("\n" + "-" * 60)
    print("测试深度学习适配器（虚拟）")
    print("-" * 60)

    dl_adapter = DeepLearningAdapter(
        model_path="dummy_model.pth",
        log_callback=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    )

    score, emotion, class_idx, probs = dl_adapter.predict(test_eeg, fs=200.0)
    print(f"预测结果: score={score:.2f}, emotion={emotion}, class_idx={class_idx}")
    print(f"概率分布: 负向={probs[0]*100:.1f}%, 中性={probs[1]*100:.1f}%, 正向={probs[2]*100:.1f}%")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
