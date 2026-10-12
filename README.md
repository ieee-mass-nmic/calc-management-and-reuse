# PEFT-Reuse

面向多配置参数高效微调选数实验的依赖感知复用与共享执行研究原型。

这是毕业论文**第二工作点**的独立项目目录。第一工作点 PCU-Select 研究“选什么数据”，本项目研究“如何保持选数行为，同时减少相关请求的重复执行”。完整研究方案、技术贡献边界、实验设计和八周计划已放入 [docs](docs/peft_reuse_execution_research_plan.md)。

**当前状态：可运行的机制验证骨架。** 默认后端使用合成特征、合成评分和简化聚类；配额及参考簇内选择函数来自冻结的 PCU-Select 源码。当前输出不能用于证明真实 PCU-Select 的质量、等价性或加速比。

## 快速运行

本机已准备好独立的 `.venv`。在项目目录运行：

```bash
cd /Users/bytedance/code/peft-reuse
.venv/bin/peft-reuse demo
```

在新环境安装（Python 3.10+，macOS / Linux）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
peft-reuse demo
```

若要复现本次开发的依赖版本，可在 Python 3.11 环境使用：

```bash
python -m pip install -c requirements-dev.lock -e ".[dev]"
```

演示生成 `2 PEFT × 2 tasks × 3 budgets = 12` 个请求，依次执行首次缓存运行、已有缓存运行、关闭磁盘缓存运行，并分别与**每个请求独立冷执行**核验。检查完整有序选集及输出中间量；失败返回非零退出码。

小规模合成计算非常便宜，磁盘写入和规划可能比计算更慢。示例验证复用边界及正确性，不预设加速收益。

## 常用命令

```bash
# 查看合并后的算子数量，并导出 JSON 计划和 Mermaid 图
.venv/bin/peft-reuse plan examples/workload.json \
  --output .runs/plan.json --mermaid .runs/plan.mmd

# 执行 + 与独立冷流程核验；输出目录自动生成
.venv/bin/peft-reuse run examples/workload.json --verify

# 同样核验，但每个共享节点重算，不读写磁盘缓存
.venv/bin/peft-reuse verify examples/workload.json --no-cache

# 测试、静态检查、来源完整性检查
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
.venv/bin/python scripts/check_reference_snapshot.py
```

`run/verify` 默认共用 `.cache/artifacts`，每次结果写入不同的 `.runs` 子目录。`--output` 可指定输出位置；已存在的输出目录或计划文件会报错，防止覆盖。每次 `demo` 创建独立缓存，因此其 cold 阶段确实从空缓存开始。

## 已实现

- 严格 JSON 请求规范化；预算显式区分 `count` 与 `fraction`，保留原始表达与解析后数量。
- 基于内容、父依赖、参数和执行上下文的产物身份；请求配置不可变快照。
- 固定评分批块、全池聚类、策略统计、全局配额与有序结果组装的依赖图。
- 跨任务、PEFT、预算请求的节点合并；命中最终结果时跳过不需要加载的祖先。
- checksum 校验、目录原子提交、单写者锁、损坏产物隔离及重算。
- 阶段事件与成本记录；失败后重新运行可以利用已完成产物。
- 独立冷执行核验、预算非嵌套反例、整簇/部分选入顺序、输入变化及故障测试。

## 项目结构

```text
peft-reuse/
├── README.md
├── pyproject.toml
├── requirements-dev.lock
├── docs/
│   ├── peft_reuse_execution_research_plan.md  # 完整研究方案
│   └── implementation.md                    # 架构、契约、接口、实施边界
├── examples/workload.json                  # 12 请求可运行样例
├── src/peft_reuse/
│   ├── schema.py / identity.py              # 规范化与内容身份
│   ├── graph.py / planner.py                # 依赖图与共享计划
│   ├── store.py / executor.py               # 可靠存储与执行
│   ├── reference.py / cli.py                # 冷执行核验与命令行
│   ├── backends/                           # 后端协议、合成后端
│   └── _reference/adaptive_quota.py         # 冻结的第一工作点函数
├── references/                             # 第一工作点材料快照、来源哈希
├── scripts/check_reference_snapshot.py
└── tests/
```

## 下一步

优先接入**固定特征快照下的真实 PCU apply 流程**：冻结模型、特征顺序、批次及数值协议，将真实评分和 MiniBatchKMeans 封装为后端，再与原 apply 的独立完整执行逐请求对照。通过这一步之后，才开始研究方案中的性能基线和输入变化实验。

真实模型后端、真实数据实验、成本驱动规划、内存释放/容量管理和评分塔共享均尚未实现。骨架中的 DAG 和缓存是研究基础设施，不能单独作为论文创新结论。

阅读：[完整研究方案](docs/peft_reuse_execution_research_plan.md) · [架构及实现说明](docs/implementation.md) · [首次验证记录](docs/validation.md) · [第一工作点材料来源](references/README.md)。

