# 固定协议的双模型读者实验

`python -m research_lab.reader_followup` 是已有 V6 检索研究的专用后续执行器。监督者指定研究问题、已有自然检索上下文、样本与预算；它自动执行协议机器审查、回答、盲评和配对统计。它不把人工指定问题描述成开放选题成功，也不自动上传论文或提交比赛。

## 输入与运行

先完成 `research_lab.reranking_study` 的可信检索项目，并保留其完整结果清单和每条上下文。配置 JSON 必须给出：

- `question`、`dataset_path`、`dataset_sha256`。
- `v6_project`、`v6_result_sha256`：已完成检索项目目录及 `artifacts/retrieval/1/result.json` 的哈希。
- `readers: ["deepseek", "kimi"]`，两个别名对应各自真实服务。
- `per_type`、`sample_seed`：STALE 两类各抽取多少场景，以固定内容哈希排序选取，不按检索或回答得分选样本。
- `model_budget` 和 `prior_charged_or_reserved`：各模型活动总额与前序已消耗/保留预留，不能重建活动清零预算。
- `max_wall_seconds`：600–86400 秒的总运行期限。
- 可选 `historical_reader_overlap`：已核对的旧读者实验重叠情况。缺失不等于从未使用。

目前协议为两种固定检索方法、1024 参考 WordPiece token、三个查询维度和两个读者。协议读取并冻结实际模型配置、调用上限与预算；服务端默认采样参数仍可能影响重复性。换服务或研究规模前应重新审查科学协议，不能宣称是完全相同的研究。

```powershell
$env:PYTHONUTF8 = "1"
$env:RESEARCH_LAB_MODELS_CONFIG = ".local/models.json"
.\.venv\Scripts\python.exe -m research_lab.reader_followup init projects/member-reader-01 --config .local/reader-config.json
.\.venv\Scripts\python.exe -m research_lab.reader_followup run projects/member-reader-01 --allow-models
```

真实调用需要团队批准额度与个人凭证。使用新项目 ID；初始化快照数据、上下文、任务和协议，绑定源码与依赖哈希。活动开始后不能直接修改输入、评审意见或源码继续实验。

## 证据和失效处理

读者只收到自然检索文本和问题，不收到新旧事实标注、排序方法名或隐藏解释。评审者收到对应维度的问题、答案和标注，方法与读者身份不作为标签传入；答案风格仍可能泄露身份。相反模型承担主评审，预选子样本使用另一模型校准一致性。模型一致不等于人工正确。

主比较按场景聚合两个模型和三个维度，做场景配对 bootstrap。缺失回答或评审在操作性主指标中记为失败，并另报完整场景敏感性和缺失值最坏界。若配置提供历史读者 ID，还单独报告排除这些 ID 的敏感性分析；这不能消除旧检索数据的重用。失败任务不选择性补跑，不能只展示成功子集。

初始化会用最长合法答案的 Unicode 和 JSON 转义边界，检查所有最坏情况下的评审消息长度；不产生 API 费用。`max_input_bytes` 是序列化 UTF-8 字节上限，`input_token_reservation` 是独立的 Token 预留额度，两者不能当成实际用量。

任务按模型分批交替执行，每批最多 3 个任务、原生并发不超过 3。请求在共享账本与活动额度中准入；未知请求保留预留。任何已存在但未完成的批次拒绝自动重复。已知发生在请求之前的启动故障也应先检查原生请求凭证，并单独留存恢复依据。

`analysis_complete` 只表示统计产物就绪，仍处于 `manuscript_pending`。论文、外部评审及比赛提交需要另行完成和验证。已结算 Token、未知用量预留、已处理任务和有效回答是不同指标。

## 输出

项目的 `reader-study-state.json` 保存阶段、额度与批次凭证；`inputs/` 保存冻结协议、样本和任务；`evidence/` 保存机器审查、回答、评审与统计。完整原始 API 回执保留在项目对应的 `outputs/v4-v7-*`。这些私有研究内容不提交公共 Git。

只读看板识别这一执行器，区分“固定协议读者实验”与“开放选题自主研究”。运行 `python -m research_lab usage-server --port 8768` 查看阶段、已处理数量和各模型额度。
