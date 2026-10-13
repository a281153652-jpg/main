# 高低压复合放电分析系统

离线读取 Dewesoft 导出的电压/电流 CSV，自动估计周期，提取动态特征并给出开路、有效放电、短路、异常放电四类**待人工复核建议**。

主程序默认运行 `dynamic-screening-v4` 动态筛查与 `batch-morphology-v3` 整批特征学习：无需人工逐周期标注，自动拟合本批次形态分布、选择群组数、给出状态建议与不确定性，并挑选群组代表周期供抽查。群组数量由 BIC 决定，群组不直接等于四类加工状态。

当前版本采用观测电平分辨尺度、噪声自适应、事件持续时间与变点形态模型，未经过真实实验标签训练。方法、参数与适用限制见 [放电状态识别方法](docs/放电状态识别方法.md)。

v4 新增两端稳定背景包络：周期前后背景电平不同，也可识别在两端背景范围之外完整起止的宽电流事件，要求同一个事件有足够的电压响应支持。背景仅用于形态分析，不从原始电流或能量中扣除。存在事件但噪声尺度仅为数值舍入下限时，SNR 输出缺失、状态保留待定，避免虚构巨大活动证据。v3 已修复的量化尺度、稀疏尖峰和极短平台问题继续保留。

无法确认的窄事件在动态筛查表中保留为异常/待复核，在自动状态表中明确记为 `undetermined`；不会自动扣除未知零电流偏置，也不会把电平间距认定为已测定的硬件 ADC 分辨率。

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

当前通过 73 项回归测试及 100 周期合成 CSV 完整流程。已上传的八个真实开发周期中，570 保留有效建议，403/600 获得新增宽事件建议，另外五个待定；只有570已由用户确认物理状态。

## 自动判别与整批学习

已有同一版本的 v3 或 v4 结果时，可独立运行整批学习，无需重读原始大 CSV。下面使用 v4 结果：

```bash
cd /workspaces/main/waveform-analysis-v2
git pull --ff-only
.venv/bin/python -m pip install --no-input -r requirements.txt
.venv/bin/python learn_batch.py "../输出结果/动态特征复核_v4/放电状态结果.csv" --output "../输出结果/批次自学习_v4"
```

新实验直接运行 `main.py` 即会提取 v4 特征并自动学习；使用 `--learning off` 可仅运行当前动态筛查。旧 v3 表仍可做原特征空间的批次学习，但不能补造 v4 背景包络或重判为 v4；同一批次不接受 v3/v4 混合表。升级完整算法须按下一节从原始 CSV 重跑。

自动学习输出：

- `自学习状态结果.csv`：`automatic_state` 是最终自动建议，`undetermined` 是证据不足；原 `state` 和 `classification_reason` 保留用于追溯输入动态模型依据。
- `波形群组统计.csv`：群组数量、代表动态特征与各自动状态数量。
- `群组代表周期.csv`：每组最接近中心和最偏离中心的周期，便于抽查。
- `自动判别统计.csv`、`自学习模型.json`：自动状态数量、学习状态、稳健缩放、BIC 比较及混合模型参数。
- `判别证据审计.csv`、`判别证据审计说明.json`：逐项检查事件持续、内部起止、振荡支持与噪声尺度，标记需要原始波形核查的特征不一致。审计不覆写状态。

整批学习使用无量纲动态特征、全协方差高斯混合模型、BIC 选组和马氏距离离群描述。默认少于 32 个有效周期时回退到动态证据建议，并明确记录 `insufficient_cycles`；常值特征不伪造群组。每个被采纳群组需至少 20 个拟合样本支持。群组归属概率不是物理状态概率，少见波形也不直接等于异常加工。

批次层 v3 在 v4 输入上增加背景包络持续时间、电压重叠及背景稳定性特征，仍保留批次层 v2 的两项策略：已有电流—电压动态证据的周期，不因统计离群单独撤销自动建议，而是提高抽查优先级；存在事件、但仅有数值舍入下限作为噪声尺度的周期不参与拟合，其状态保留待定并记录 `untrusted_numerical_noise_floor`。数字常值零信号不会仅因噪声标签为 `numerical_floor` 被排除。

物理状态名称仍由逐周期电流—电压动态证据支持；自学习层不把证据不足的窄事件自动变成开路或有效放电，不根据群组大小强行产生四类。当前没有训练好的四类监督分类器，也没有独立实验准确率。人工用于抽查和验证，而非运行前逐周期填写标签。自学习表中的 `review_required` 是抽查调度标记：高优先级周期需复核，典型支持周期可抽查；原标记保留为 `dynamic_review_required`。`automatic_label_validated=False` 明确表示自动建议尚未完成独立物理验证。

## 从现有 v3 结果升级到 v4

在此前的 `/workspaces/main/waveform-analysis-v2` 终端中执行：

```bash
cd /workspaces/main/waveform-analysis-v2
git pull --ff-only
.venv/bin/python -m pip install --no-input -r requirements.txt
.venv/bin/python main.py "../输入数据/gyb1-4.csv" --cycles "../输出结果/动态特征复核_v3/周期信息.csv" --expected-source-sha256 6faa67c6c37dedd0d39f4ce039c0bff9bdcb84baef3b88feeb06b429a313f047 --output "../输出结果/动态特征复核_v4" --examples 10
```

这里复用**同一个 CSV** 的旧自动检测边界，保持周期编号一致，不要求手工指定周期长度。上面的 SHA-256 来自该实验导出来源记录，只适用于这份 `gyb1-4.csv`；其他实验应使用各自来源记录或省略该选项并重新自动检测周期。主程序在分析前核对文件哈希，不匹配时停止且不覆盖结果。`运行来源.json` 记录实际哈希、预期哈希、点数、时间范围和边界来源。

v3 特征缺少两端背景包络，不能只修改旧 CSV 的模型版本或状态列来代替 v4 原始波形提取。尚未在本地取得完整原始实验 CSV，因此没有报告 v4 的全批次类别数量或实验准确率。周期 570 已由用户确认有效加工放电；403、600 等新规则输出仍仅为候选，不作为独立真值。

## 导出少量原始周期复核

```bash
.venv/bin/python export_review_cycles.py --results "../输出结果/动态特征复核_v4/放电状态结果.csv" --ids 22,359,403,600
```

导出脚本默认选择原 v2 结果中的 29、103、570、605 周期，将完整原始 `time/voltage/current`、`cycle_id/source_index` 和来源映射写入 `../输出结果/周期复核包/批次目录/`。导出不平滑、不施加额外零点修正、不重新定位边界。请上传较小的 `周期复核样本.csv` 及同目录的来源 JSON 以进行逐点复核。可用 `--ids`、`--results`、`--output` 改变选择。
