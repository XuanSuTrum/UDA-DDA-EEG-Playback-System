# UDA-DDA 模型输出接入与 SEED 上位机离线回放流程

## 1. 离线模型阶段

UDA-DDA 的训练、无监督域适应、模型选择和权重保存均在离线阶段完成。仓库不分发训练数据、被试级特征、scaler 或模型权重。

## 2. 正式预测生成

正式输出固定使用目标被试前 3 个 trial 的校准特征拟合 scaler：

```text
scaler_mode = calibration_feature
scaler_postprocess = clip
```

trial 4 至 trial 15 的正式回放数据不参与 scaler 拟合。真实标签不参与 scaler 选择、后处理参数选择或默认输出文件选择。

温度 `3.0` 和尾随窗口长度 `10` 必须在正式回放前固定。温度校准只生成显示概率，原始三分类 `prob_*` 字段不变。

## 3. 因果显示后处理

对每个 trial 分别处理，并在新 trial 开始时重置历史：

```text
display_prob_negative(k)
  = median(prob_negative_calibrated[max(1, k-9) ... k])
```

约束：

- 只使用当前及历史窗口；
- 不允许使用未来窗口；
- `rolling(center=False)`；
- 不跨 trial 平滑；
- 不对缺失窗口插值；
- trial 级均值、中位数或多数投票不参与当前显示状态。

逐窗口定义：

```text
display_prob_non_negative = 1 - display_prob_negative
display_negative_score = 100 * display_prob_negative
display_state = 负性 if display_prob_negative >= 0.5 else 非负性
```

trial 摘要只在回放结束后用于描述性评估。

## 4. 泄漏诊断隔离

`match_training_test` 使用回放/测试集统计量，属于 `leaky_diagnostic`。默认不执行，仅在传入 `--include-leaky-diagnostic` 时运行。

启用后：

- 打印明确警告；
- 输出文件名包含 `diagnostic_only`；
- 不复制为默认预测文件；
- 不用于部署或正式显示。

## 5. 上位机离线回放

`app/eeg_viewer2.py` 读取回放脑电文件和同目录的 `predictions_display.csv`。启用 `prediction_mode_enabled=True` 后：

- 按 `time_sec` 逐窗口读取因果 `display_state`；
- 不调用 `model_thread.do_inference`；
- 原始标注只用于验证显示和日志；
- 右上角保持“当前情绪状态输出”信息卡片；
- 不恢复历史概率小曲线；
- 波形与频谱回放保持不变。

## 6. 日志绘图

日志绘图脚本只解析实际存在的 `[预测同步]` 行。蓝线表示 `P(negative)`，背景表示原始 trial 标注，50% 虚线仅表示二类等概率参考位置。不同 trial 不连接，缺失时间点不插值，不显示网格线。

## 7. 验证边界

本仓库展示预计算模型结果接入上位机离线回放，不代表实时采集期间模型推理，也不构成自主刺激控制验证。
