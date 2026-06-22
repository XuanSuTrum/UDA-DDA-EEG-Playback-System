# UDA-DDA 模型输出接入与 SEED 上位机离线回放验证流程

## 1. 离线模型阶段

UDA-DDA 的训练、无监督域适应、模型选择和权重保存均在离线阶段完成。本仓库只保留用于公开展示的模型结构和离线预测结果生成脚本，不分发训练数据、被试级特征文件、scaler 或模型权重。

## 2. 后台预测结果生成

`inference/generate_upper_demo_predictions_lds.py` 读取已生成的 DE+LDS 特征、回放元数据和 UDA-DDA 权重，输出带 `time_sec` 的窗口级预测结果。

关键约定：

- `prob_negative` 是三分类 softmax 中 negative 类概率。
- `prob_non_negative = prob_neutral + prob_positive`。
- `display_prob_negative` 是上位机用于显示的窗口或片段级消极概率。
- `display_prob_non_negative = 1 - display_prob_negative`。
- `true_label_name` 只用于离线验证显示和日志对照，不参与概率计算。

## 3. 上位机离线回放

`app/eeg_viewer2.py` 读取回放脑电文件，并在同目录查找 `subject15_trial4_15_predictions_display.csv` 等预测文件。找到预测文件后会启用 `prediction_mode_enabled=True`。

该模式下：

- 上位机只根据 `predictions_display.csv` 的 `time_sec` 同步显示预测状态。
- 不在回放线程中调用 UDA-DDA/SVM/EEGNet 实时推理。
- 不调用 `model_thread.do_inference`。
- MAT/CSV 回放数据用于波形和频谱显示；预测状态来自 CSV。

## 4. 日志绘图

`analysis/plot_upper_demo_negative_history_from_log.py` 解析上位机日志中的 `[预测同步]` 行，生成消极类别概率连续输出图。

绘图语义：

- 蓝线表示 `P(negative)`。
- 背景色表示 SEED 原始试次标注。
- 50% 虚线表示二类等概率参考线，不是刺激触发阈值。
- 不连接不同 trial 之间的曲线。
- 缺失日志时间点不插值。
- 不显示背景网格线。

## 5. 当前验证边界

本仓库展示的是“离线预测结果接入上位机回放”的验证链路，不代表实时脑电采集验证，不代表回放过程中实时调用模型，也不构成自主闭环刺激控制验证。
