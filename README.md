# 高低压复合放电分析系统

离线读取 Dewesoft 导出的电压/电流 CSV，自动估计周期，提取动态特征并给出开路、有效放电、短路、异常放电四类**待人工复核建议**。

本版本采用噪声自适应与变点/振荡形态模型，未经过真实实验标签训练。方法、参数与适用限制见 [放电状态识别方法](docs/放电状态识别方法.md)。

## 安装与运行

Python 3.12，使用现有仓库目录，无需另建 Git worktree。

```bash
cd /workspace/main
python -m venv .venv
.venv/bin/python -m pip install --no-input -r requirements.txt
.venv/bin/python main.py "输入数据/gyb1-4.csv" --output "输出结果" --examples 10
```

Windows 可用 `py -3.12 -m venv .venv`，再用 `.venv\Scripts\python.exe` 运行 pip 与 `main.py`。

默认输入为 `输入数据/gyb1-4.csv`；也可以指定其他路径。没有真实 CSV 时程序会明确报告输入缺失，不会用合成数据代替实验结果。时间须为秒，电压/电流须已正确标定到 V/A，识别器接收 `time/voltage/current` 标准列。

如已从确认无电流区测得零电流偏置，可额外传 `--current-offset 标定值`（安培）。默认 0，不自动猜测偏置。

## 输出与复核

输出包含状态结果、动态特征统计、短路候选和波形图。`状态验证/批次目录/人工复核清单.csv` 提供每类默认 10 张样本图的标签填写入口；样本不足会明确报告。波形采用蓝色电压、橙色有符号电流的上下子图。

特征提取保留 `extract_features(cycle)`、`classify_state(features)` 和 `recognize_discharge_states(data, cycles, output_dir)` 接口；可通过可选 `DynamicConfig` 配置统计/形态参数与零点校准。异常现在单独输出为 `abnormal_discharge`。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试使用合成数据验证软件行为，不能代替真实实验波形验证或作为论文中的识别准确率。
