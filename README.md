# 能工智人 · BDCI Auto Research

基于 openJiuwen / JiuwenSwarm 的科研框架，面向 [CCF BDCI 1167](https://www.xir.cn/competition/1167)。版本 **0.8.0** 支持受监督项目、限定领域自主科研，以及**从宽泛方向自动选题的科研闭环**。

成员可以在各自电脑克隆代码，配置自己的模型 API Key 与预算，独立运行项目，通过 PR 合作开发。开放选题模式接收宽泛目标、数据资产和预算，检索文献、提出多个问题、审查并选择课题，再运行实验和写作。执行能力目前覆盖 STALE 检索审计与数值表格分类；超出能力会换题或停止。它保留失败和负结果，不保证创新性或发表质量，也不自动上传论文或提交比赛。

## 从这里开始

- [开放选题自主科研](docs/OPEN_TOPIC_RESEARCH.md)：宽泛方向 → 文献 → 课题池 → 可行性与双审查 → 实验和论文；含离线验收与真实配置。
- [自主科研运行指南](docs/AUTONOMOUS_RESEARCH.md)：无人逐阶段审批的完整闭环、可执行范围、预算与故障恢复。
- [队友安装和运行指南](docs/TEAM_HANDOFF.md)：从零安装、个人模型配置、独立研究、论文与材料交接。
- [新课题开发指南](docs/NEW_RESEARCH.md)：复现已有课题和开展新研究的区别，可信适配器接口。
- [本地学习式重排研究](docs/LOCAL_RERANKING.md)：固定候选、参考 token 预算与配对统计；无需 API Key。
- [固定协议读者实验](docs/READER_FOLLOWUP.md)：在已验证的自然检索上下文上比较两个模型的实际回答，保留盲评与缺失结果。
- [协作和候选作品选择](docs/COLLABORATION.md)：分支、预算分配、冻结实验、评审与统一提交。
- [框架契约](docs/framework-v5-reference.md)：项目阶段、监督决策、恢复、预算、数据与执行边界。
- [第三方来源](THIRD_PARTY.md) · [验证记录](docs/VALIDATION.md)。

## 快速安装（Windows / PowerShell，Python 3.12 + Git）

```powershell
git clone https://github.com/YidaYang/bdci.git
cd bdci
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts/bootstrap.py
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m research_lab project doctor
.\.venv\Scripts\python.exe scripts/verify_projects_v5.py --output projects/my-first-check
```

这段流程下载固定版本依赖，执行两个确定性示例，**不调用模型、不产生真实论文**。最后一条命令请使用尚不存在的目录。Linux 使用 `python3.12 -m venv .venv`，后续将 Python 路径替换为 `.venv/bin/python`。

## 配置自己的模型

先向队长确认预算。以下每模型 5M 仅为配置示例，不代表团队批准的配额：

```powershell
.\.venv\Scripts\python.exe scripts/setup_member.py --member alice --deepseek-budget 5000000 --kimi-budget 5000000
```

编辑 `.local/models.json` 中的 `base_url`、模型名、接口格式；编辑 `.env.local` 填入自己的 Key。默认地址 `127.0.0.1:8080` 是占位值。两个别名 `deepseek`、`kimi` 对应执行者和规划/评审者，可以配置成兼容服务商提供的模型；更换模型后要记录实际型号并验证输出契约。

```powershell
# 只显示账本和看板，不发起模型请求
.\.venv\Scripts\python.exe -m research_lab usage
.\.venv\Scripts\python.exe -m research_lab usage-server --port 8768
```

打开 <http://127.0.0.1:8768>。个人配置、密钥、账本、数据、运行产物、论文和比赛包全部被 Git 忽略。账本是本机范围，不会自动合并五人的预算。

## 开始一份独立研究

支持两种入口：

1. 通用项目：`python -m research_lab project init projects/alice-topic --config <项目配置>`，由监督者选择和注册可信实验适配器。
2. STALE 复现起点：`python -m research_lab.study --id alice-stale-01 --dataset data/stale/T1_T2_400_FULL.json`。先按交接指南下载数据；默认每类 2 个读者样本，仅用于小规模试运行。检索仍覆盖输入数据集。它复用已有方法，不能直接作为新发现。

随后统一使用 `project advance/status/review/decide`。普通 `advance` 不调用模型；真实调用必须加 `--allow-models`，且关键阶段仍需监督者阅读产物并验收。

## 仓库边界

这里提供可继续开发的源码、测试、公开配置模板和上游补丁。2026-10-01 提交的 V4 比赛包及 V4/V5 原始科研记录保留在队长的冻结档案中，未作为公共仓库内容上传。框架版本与论文版本是两套编号。

JiuwenSwarm 上游贡献补丁位于 `integrations/jiuwenswarm/contribution_v4_with_tests.patch`，公开上游 PR 仍需团队按比赛要求自行提交。本仓库的协作 PR 不等于上游贡献 PR。
