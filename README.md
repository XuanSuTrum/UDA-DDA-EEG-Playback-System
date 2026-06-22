# UDA-DDA EEG Playback System

基于 UDA-DDA（Unsupervised Domain Adaptation with Dynamic Distribution Alignment）模型输出的 SEED 上位机离线回放验证项目。仓库整理的是“后台预测结果生成 -> 上位机按时间戳同步显示 -> 日志解析绘图”的公开展示链路。

## 系统定位

当前系统的准确定位是：

1. UDA-DDA 模型训练和无监督域适应在离线阶段完成。
2. 后台推理脚本生成带时间戳的窗口级预测结果。
3. 上位机读取脑电回放数据和 `predictions_display.csv`。
4. 上位机按照时间戳同步显示波形、频谱、概率、状态和日志。
5. 独立绘图脚本解析日志并生成消极类别概率连续曲线。

当前不是实时脑电采集验证，不是在上位机回放过程中实时调用模型，也不构成自主刺激控制验证。

## 工作流程

```text
离线训练/域适应
  -> DE+LDS 特征和 UDA-DDA 权重
  -> inference/generate_upper_demo_predictions_lds.py
  -> subject15_trial4_15_predictions_display.csv
  -> app/eeg_viewer2.py 离线回放同步显示
  -> analysis/plot_upper_demo_negative_history_from_log.py 生成概率曲线
```

详见 [docs/workflow.md](docs/workflow.md)。

## 仓库结构

```text
.
├── app/
│   ├── eeg_viewer2.py
│   └── model_adapters.py
├── inference/
│   ├── generate_upper_demo_predictions_lds.py
│   ├── test_offline_lds.py
│   ├── SDA_DDA.py
│   ├── backbone.py
│   ├── mmd.py
│   ├── cmmd.py
│   └── utils.py
├── analysis/
│   └── plot_upper_demo_negative_history_from_log.py
├── configs/
│   └── config.example.yaml
├── examples/
│   ├── predictions_display.example.csv
│   ├── upper_demo_prediction_sync.example.log
│   └── negative_history.example.png
├── docs/
│   └── workflow.md
└── tests/
    └── test_prediction_artifacts.py
```

## 环境安装

建议使用 Python 3.9 或更高版本。当前整理过程使用本地 Python 环境完成语法检查和轻量测试。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS 激活命令为 `source .venv/bin/activate`。

## 后台预测结果生成

SEED 原始数据、被试级特征文件和模型权重不随仓库分发。准备好本地文件后运行：

```bash
python inference/generate_upper_demo_predictions_lds.py ^
  --calib-feature data/seed_features/15_calibration_3trials.mat ^
  --online-feature data/seed_features/15_online_remaining_trials.mat ^
  --model-path models/subject15_calib_supervised_best_model.pth ^
  --meta-csv data/upper_demo/subject15_trial4_15_replay_meta.csv ^
  --output-dir outputs/upper_demo
```

脚本会生成 `subject15_trial4_15_predictions_display.csv`。上位机离线回放时，该 CSV 是预测结果来源。

## 上位机启动

```bash
python app/eeg_viewer2.py
```

在界面中加载回放 MAT/CSV 文件。上位机会在回放文件同目录查找 `subject15_trial4_15_predictions_display.csv` 等候选预测文件。找到后启用 `prediction_mode_enabled=True`，此时不会实时运行 UDA-DDA/SVM/EEGNet 推理，也不会调用 `model_thread.do_inference`。

右上角信息卡片显示：当前 Trial、原始试次标注、预测状态、消极类别概率、非消极类别概率、消极情绪得分和预测来源。原始试次标注仅用于离线验证显示，不参与概率计算。

## 日志绘图

```bash
python analysis/plot_upper_demo_negative_history_from_log.py ^
  --log examples/upper_demo_prediction_sync.example.log ^
  --output outputs/negative_history_from_log.png
```

示例输出图：

![negative history example](examples/negative_history.example.png)

绘图约定：蓝线表示 `P(negative)`；背景表示 SEED 原始试次标注；50% 虚线仅为二类等概率参考线，不是刺激触发阈值；不同 trial 之间不连线；缺失日志时间点不插值；不显示背景网格线。

## predictions_display.csv 字段

示例见 [examples/predictions_display.example.csv](examples/predictions_display.example.csv)。核心字段包括：

- `time_sec`：窗口中心对应的回放时间戳。
- `trial_id`、`trial_time_sec`：SEED 试次编号和试次内时间。
- `true_label_name`、`raw_label`：原始试次标注，仅用于离线验证显示。
- `prob_negative`、`prob_neutral`、`prob_positive`：UDA-DDA 三分类概率。
- `prob_non_negative`：`P(neutral) + P(positive)`。
- `display_state`：上位机展示的二类状态。
- `display_prob_negative`、`display_prob_non_negative`、`display_negative_score`：上位机同步显示字段。
- `feature_source`、`scaler_mode`、`display_probability_source`：预测来源和后处理说明。

概率定义：

```text
P(non-negative) = P(neutral) + P(positive) = 1 - P(negative)
```

## 数据与边界

本仓库不包含 SEED 原始数据、被试级 MAT/EDF/BDF/SET/NPY/NPZ 文件、模型权重、训练检查点、scaler PKL、完整 `predictions_display.csv` 或完整系统运行日志。使用者需按 SEED 数据集许可自行获取数据，并在本地未跟踪目录中配置路径。

本项目只展示离线回放验证链路，不宣称实时在线模型部署，也不宣称已实现自主闭环刺激控制。
