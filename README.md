# 高低压复合放电分析系统

离线读取 Dewesoft 导出的电压/电流 CSV，自动估计周期，提取动态特征并给出开路、有效放电、短路、异常放电四类**待人工复核建议**。

当前版本为 `dynamic-screening-v3`，采用观测电平分辨尺度、噪声自适应、事件持续时间与变点形态模型，未经过真实实验标签训练。方法、参数与适用限制见 [放电状态识别方法](docs/放电状态识别方法.md)。

v3 修复量化平台导致 MAD 噪声退化、稀疏尖峰被当作持续振荡、极短后段标准差被记为 0 的问题。无法确认的量化窄事件输出异常/待复核；不会自动扣除未知零电流偏置，也不会把电平间距认定为已经测定的硬件 ADC 分辨率。

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

## 在已运行 v2 的代码空间更新与复核

在此前的 `/workspaces/main/waveform-analysis-v2` 终端中执行：

```bash
git pull --ff-only
.venv/bin/python main.py "../输入数据/gyb1-4.csv" --cycles "../输出结果/动态特征复核/周期信息.csv" --output "../输出结果/动态特征复核_v3" --examples 10
.venv/bin/python export_review_cycles.py
```

这里复用**同一个 CSV** 的自动检测边界，保持周期编号一致以比较算法。不同实验文件不能直接复用周期表。v3 保存 `运行来源.json` 的输入 SHA-256 以便追溯；v2 特征表缺少新增事件信息，不能仅修改旧 CSV 的状态列来代替重新提取。

导出脚本默认选择原 v2 结果中的 29、103、570、605 周期，将完整原始 `time/voltage/current`、`cycle_id/source_index` 和来源映射写入 `../输出结果/周期复核包/批次目录/`。导出不平滑、不施加额外零点修正、不重新定位边界。请上传较小的 `周期复核样本.csv` 及同目录的来源 JSON 以进行逐点复核。可用 `--ids`、`--results`、`--output` 改变选择。
