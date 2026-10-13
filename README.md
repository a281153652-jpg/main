# 高低压复合放电分析系统

离线读取 Dewesoft 导出的电压/电流 CSV，自动估计周期，提取动态特征并给出开路、有效放电、短路、异常放电四类**待人工复核建议**。

主程序现在默认增加 `batch-morphology-v2` 整批特征学习：无需人工逐周期标注，自动拟合本批次形态分布、选择群组数、给出状态建议与不确定性，并挑选群组代表周期供抽查。群组数量由 BIC 决定，群组不直接等于四类加工状态。

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

## 自动判别与整批学习

已有 v3 运行结果时，可在代码空间终端执行以下命令，无需重读原始大 CSV：

```bash
cd /workspaces/main/waveform-analysis-v2
git pull --ff-only
.venv/bin/python -m pip install --no-input -r requirements.txt
.venv/bin/python learn_batch.py "../输出结果/动态特征复核_v3/放电状态结果.csv" --output "../输出结果/批次自学习"
```

新实验直接运行 `main.py` 即会提取特征并自动学习；使用 `--learning off` 可仅运行原 v3 筛查以作对照。

自动学习输出：

- `自学习状态结果.csv`：`automatic_state` 是最终自动建议，`undetermined` 是证据不足；原 `state` 和 `classification_reason` 保留用于追溯 v3 依据。
- `波形群组统计.csv`：群组数量、代表动态特征与各自动状态数量。
- `群组代表周期.csv`：每组最接近中心和最偏离中心的周期，便于抽查。
- `自动判别统计.csv`、`自学习模型.json`：自动状态数量、学习状态、稳健缩放、BIC 比较及混合模型参数。
- `判别证据审计.csv`、`判别证据审计说明.json`：逐项检查事件持续、内部起止、振荡支持与噪声尺度，标记需要原始波形核查的特征不一致。审计不覆写状态。

整批学习使用无量纲动态特征、全协方差高斯混合模型、BIC 选组和马氏距离离群描述。默认少于 32 个有效周期时回退到动态证据建议，并明确记录 `insufficient_cycles`；常值特征不伪造群组。每个被采纳群组需至少 20 个拟合样本支持。群组归属概率不是物理状态概率，少见波形也不直接等于异常加工。

批次层 v2 修正两项策略：已有电流—电压动态证据的周期，不因统计离群单独撤销自动建议，而是提高抽查优先级；存在事件、但仅有数值舍入下限作为噪声尺度的周期不参与拟合，其状态保留待定并记录 `untrusted_numerical_noise_floor`。数字常值零信号不会仅因噪声标签为 numerical_floor 被排除。原逐周期特征仍是 dynamic-screening-v3，升级批次策略不需要重新读取大波形文件。

物理状态名称仍由逐周期电流—电压动态证据支持；自学习层不把证据不足的窄事件自动变成开路或有效放电，不根据群组大小强行产生四类。当前没有训练好的四类监督分类器，也没有独立实验准确率。人工用于抽查和验证，而非运行前逐周期填写标签。自学习表中的 `review_required` 是抽查调度标记：高优先级周期需复核，典型支持周期可抽查；原标记保留为 `dynamic_review_required`。`automatic_label_validated=False` 明确表示自动建议尚未完成独立物理验证。

## 在已运行 v2 的代码空间更新与复核

在此前的 `/workspaces/main/waveform-analysis-v2` 终端中执行：

```bash
git pull --ff-only
.venv/bin/python main.py "../输入数据/gyb1-4.csv" --cycles "../输出结果/动态特征复核/周期信息.csv" --output "../输出结果/动态特征复核_v3" --examples 10
.venv/bin/python export_review_cycles.py
```

这里复用**同一个 CSV** 的自动检测边界，保持周期编号一致以比较算法。不同实验文件不能直接复用周期表。v3 保存 `运行来源.json` 的输入 SHA-256 以便追溯；v2 特征表缺少新增事件信息，不能仅修改旧 CSV 的状态列来代替重新提取。

导出脚本默认选择原 v2 结果中的 29、103、570、605 周期，将完整原始 `time/voltage/current`、`cycle_id/source_index` 和来源映射写入 `../输出结果/周期复核包/批次目录/`。导出不平滑、不施加额外零点修正、不重新定位边界。请上传较小的 `周期复核样本.csv` 及同目录的来源 JSON 以进行逐点复核。可用 `--ids`、`--results`、`--output` 改变选择。
