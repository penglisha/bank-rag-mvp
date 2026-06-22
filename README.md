# 银行合规知识助手 — RAG MVP

基于检索增强生成（RAG）技术的银行合规问答系统，用于学习 RAG 核心原理。

**完全本地可运行**：Embedding 使用本地 sentence-transformers，LLM 支持本地 Ollama，无需任何 API Key 也能跑通完整流程。

---

## 快速开始

### 1. 安装依赖

```bash
cd bank-rag-mvp
pip install -r requirements.txt
```

### 2. 配置 .env

```bash
cp .env.example .env
```

根据你使用的 LLM 后端，选择其中一种配置方式：

**方式 A：Ollama（本地免费，推荐）**

```bash
# 1. 安装 Ollama：https://ollama.com/download
# 2. 拉取中文模型
ollama pull qwen2.5:7b
```

`.env` 中设置：
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:7b
```

**方式 B：Anthropic Claude API**

`.env` 中设置：
```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=your_key_here    # 从 console.anthropic.com 获取
ANTHROPIC_MODEL=claude-haiku-4-5
```

### 3. 运行

```bash
# 确保 Ollama 服务在运行（方式 A 需要）
ollama serve

# 启动问答系统（新开一个终端）
python src/main.py
```

首次运行会自动切块文档、下载 Embedding 模型（约 400MB）并构建向量库，后续启动直接复用。

```bash
python src/main.py --rebuild    # 修改 docs/ 文档后重建向量库
```

---

## RAG 流程数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                   【构建阶段】（首次运行，后续跳过）          │
│                                                             │
│  docs/*.txt                                                 │
│      │                                                      │
│      ▼  chunking.py  按段落切块（chunk_size=400，overlap=50）│
│  文本块列表 [{text, metadata}, ...]   共约 55 块             │
│      │                                                      │
│      ▼  embedding.py  本地 sentence-transformers 向量化      │
│  向量列表 [[0.12, -0.34, ...], ...]   每块 384 维            │
│      │                                                      │
│      ▼  vectorstore.py  写入 Chroma 本地向量库               │
│  chroma_db/  （向量 + 原文 + 元数据，持久化到本地文件）       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   【查询阶段】（每次提问）                    │
│                                                             │
│  用户输入："信用卡逾期30天会有什么后果？"                     │
│      │                                                      │
│      ▼  embedding.py  同一模型向量化查询                     │
│  查询向量 [0.08, -0.29, ...]                                │
│      │                                                      │
│      ▼  vectorstore.py  余弦相似度搜索 top-k                 │
│  相关文本块（附相似度分数）：                                 │
│  ├── 《信用卡逾期处理规定》第10条...  相似度 0.71             │
│  ├── 《信用卡逾期处理规定》第4.1款... 相似度 0.61             │
│  └── 《个人贷款产品说明》第9.3款...  相似度 0.62             │
│      │                                                      │
│      ▼  generate.py  拼 Prompt → 调用 LLM                   │
│      │                                                      │
│      ├── LLM_BACKEND=ollama    → 本地 qwen2.5:7b（免费）    │
│      └── LLM_BACKEND=anthropic → Anthropic Claude API       │
│                                                             │
│  带引用来源的最终回答：                                      │
│  "根据《信用卡逾期处理规定》第10.1款，逾期31-60天移交        │
│   催收专员处理；第10.4款规定超90天上报征信系统..."           │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
bank-rag-mvp/
├── docs/                       # 银行合规文档（原始知识库）
│   ├── 个人贷款产品说明.txt
│   ├── 信用卡逾期处理规定.txt
│   ├── 理财产品风险揭示书.txt
│   ├── 个人征信查询规定.txt
│   └── 客户投诉处理流程.txt
├── src/
│   ├── chunking.py             # 文档切块（段落优先 + overlap）
│   ├── embedding.py            # 文本向量化（本地 / OpenAI 双模式）
│   ├── vectorstore.py          # 向量存储与检索（Chroma）
│   ├── generate.py             # LLM 生成回答（Ollama / Claude 双后端）
│   ├── main.py                 # 纯 RAG 入口（保持不变）
│   ├── tools.py                # Agent 工具：客户查询 + 操作申请
│   ├── router.py               # 意图路由（json_parse / tool_calling 双模式）
│   └── agent_main.py           # Agent 入口（意图路由 + 工具调用）
├── chroma_db/                  # 向量库（自动生成，删除后 --rebuild 重建）
├── test_questions.txt          # 11 道测试题 + 结果记录表格
├── .env                        # 你的配置（不要提交到 git）
├── .env.example                # 配置模板
└── requirements.txt
```

---

## 可调参数

在 `.env` 中修改以下参数并重新运行，观察效果变化：

### LLM 后端

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `LLM_BACKEND` | `ollama`（默认）/ `anthropic` | 切换本地 Ollama 或 Anthropic Claude |
| `OLLAMA_MODEL` | `qwen2.5:7b`（默认）等 | Ollama 使用的模型，需提前 `ollama pull` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5`（默认）等 | Anthropic 使用的模型 |

### Embedding 模式

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODE` | `local`（默认）/ `openai` | 本地免费 vs OpenAI 付费（维度 384 vs 1536） |

> 切换 Embedding 模式后必须 `--rebuild`，否则查询向量和存储向量维度不一致。

### RAG 参数

| 参数 | 默认值 | 调小效果 | 调大效果 |
|------|--------|----------|----------|
| `CHUNK_SIZE` | 400 | 检索更精准，但单块上下文少 | 每块信息更完整，但语义不够聚焦 |
| `TOP_K` | 3 | Prompt 更短，可能遗漏相关条款 | 上下文更丰富，但噪音增多 |

---

## 推荐实验

### RAG 基础实验
1. **改变 CHUNK_SIZE**：分别设置 200 / 400 / 800，问同一个问题，对比检索片段的精准度
2. **改变 TOP_K**：设置 1 vs 5，观察回答完整性的差异
3. **切换 LLM 后端**：同一问题分别用 Ollama 和 Claude 回答，对比质量
4. **切换 EMBEDDING_MODE**：对比 local 和 openai 的相似度分数差异（需 OpenAI Key + `--rebuild`）
5. **添加新文档**：在 `docs/` 放入新 .txt 文件，`python src/main.py --rebuild` 重建

### Agent 路由实验
6. **跑测试题集**：用 `test_questions.txt` 的 11 道题，分别在两种路由模式下测试，填写结果记录表
7. **切换路由模式对比**：`ROUTING_MODE=tool_calling python src/agent_main.py` vs 默认 json_parse
8. **换路由模型**：`ROUTER_MODEL=gemma4-hermes:latest`，对比 tool_calling 成功率

---

## Agent 模式说明

### 运行方式

```bash
# 默认 json_parse 路由
python src/agent_main.py

# 切换到 tool_calling 路由
ROUTING_MODE=tool_calling python src/agent_main.py

# 换路由模型（gemma4-hermes 对 tool calling 支持更好）
ROUTER_MODEL=gemma4-hermes:latest ROUTING_MODE=tool_calling python src/agent_main.py
```

### 两种路由方式原理对比

| 维度 | json_parse（默认） | tool_calling |
|------|-------------------|--------------|
| **原理** | Prompt 要求模型只输出 JSON，Python 解析 intent + params | 发送工具定义，模型自主选择调用哪个工具及参数 |
| **Ollama API** | 原生 `/api/chat` + `format:"json"` | OpenAI 兼容端点 + `tools` 参数 |
| **稳定性** | 高（format:json 保证合法 JSON） | 依赖模型是否经过工具调用微调 |
| **参数提取** | 依赖 prompt 设计，偶尔提取不完整 | 通常更准确，模型直接生成结构化参数 |
| **调试难度** | 低（直接看 prompt 和 JSON 输出） | 略高（需理解 tool_calls 结构） |
| **模型要求** | 几乎所有模型都能跟 JSON 指令 | 需要 function calling 微调（hermes3、qwen2.5 等）|
| **适用场景** | 本地小模型首选；快速原型 | 云端 API（OpenAI/Claude）；支持 FC 的本地模型 |

**结论：本地小模型（7B 以下）优先用 json_parse**。tool_calling 在 qwen2.5:7b 上实测发现，5/9 道有效题触发了降级（模型忽略 `tool_choice="required"` 直接回答），但降级链路能兜住，最终正确率不受影响。如果追求 tool_calling 原生成功率，换用 `gemma4-hermes:latest`。

### 测试结果（qwen2.5:7b 实测）

> 运行方式：`python run_tests.py`

| 题号 | 预期意图 | json_parse 判断 | 正确？ | tool_calling 判断 | 正确？ |
|------|----------|----------------|--------|------------------|--------|
| Q01  | 知识型   | 知识型          | ✅     | 知识型（原生）     | ✅     |
| Q02  | 知识型   | 知识型          | ✅     | 知识型（原生）     | ✅     |
| Q03  | 知识型   | 知识型          | ✅     | 知识型（原生）     | ✅     |
| Q04  | 知识型   | 知识型          | ✅     | 知识型（↩降级）    | ✅     |
| Q05  | 查询型   | 查询型          | ✅     | 查询型（原生）     | ✅     |
| Q06  | 查询型   | 查询型          | ✅     | 查询型（↩降级）    | ✅     |
| Q07  | 查询型   | 查询型          | ✅     | 查询型（↩降级）    | ✅     |
| Q08  | 操作型   | 操作型          | ✅     | 操作型（↩降级）    | ✅     |
| Q09  | 操作型   | 操作型          | ✅     | 操作型（原生）     | ✅     |
| Q10  | 模糊     | 知识型（↩降级） | —      | 查询型（原生）     | —      |
| Q11  | 模糊     | 操作型          | —      | 操作型（原生）     | —      |
| **合计** | — | — | **9/9 (100%)** | — | **9/9 (100%)** |

> ↩降级 = tool_calling 未调用工具，自动回退到 json_parse 兜底
> Q10/Q11 为模糊题，不计入正确率；两种模式对模糊题的处理方式有所不同（见下方分析）

### 操作型为什么"生成申请而不直接执行"

金融场景下，所有涉及账户状态变更或资金流转的操作，都必须经过人工复核——这是监管合规的硬性要求，也是风控的最后一道防线。

**Human-in-the-loop 设计的意义：**
- **防止误操作**：AI 意图识别可能出错（小概率），操作型问题一旦执行后果难以回滚
- **满足审批留痕**：金融业务要求全程可审计，申请单自动留存操作记录
- **责任归属清晰**：AI 只负责"信息整理和申请生成"，人工负责"最终决策"
- **符合 HKMA/银保监会要求**：监管明确要求 AI 辅助而非替代关键业务决策

在代码中体现为：`submit_operation_request` 只打印申请单、返回申请编号，不修改任何 `CUSTOMER_DB` 数据。

---

## 常见问题

**Q: 首次运行很慢**
A: 正在下载本地 Embedding 模型（paraphrase-multilingual-MiniLM-L12-v2，约 400MB），只需下载一次。

**Q: 用 Ollama 时报 `Connection error`**
A: 两种原因：① Ollama 服务未启动，运行 `ollama serve`；② 系统代理拦截了 localhost 请求，代码已通过 `httpx.Client(trust_env=False)` 处理，如仍报错请检查代理配置。

**Q: 修改了文档，回答没有更新**
A: 向量库首次构建后会缓存。修改 `docs/` 后需运行 `python src/main.py --rebuild`。

**Q: 切换了 EMBEDDING_MODE 后报维度错误**
A: 不同模型输出的向量维度不同（local=384，openai=1536），混用会报错。切换后必须 `--rebuild`。

**Q: tool_calling 模式下知识型问题走了 json_parse 兜底**
A: qwen2.5:7b 对纯知识型问题有时会直接回答而不调用工具（即使设置了 `tool_choice="required"`）。这是模型行为，代码已做降级处理。换用 `ROUTER_MODEL=gemma4-hermes:latest` 可改善。

**Q: 想用其他 Ollama 模型路由**
A: `ollama pull <model>` 后，`ROUTER_MODEL=<model> python src/agent_main.py`。推荐 tool_calling 场景用 `gemma4-hermes:latest`（本机已有）。

---

## 自动化评估系统

### 运行方式

```bash
python run_eval.py                  # 运行全部三个评估模块
python run_eval.py --no-quality     # 跳过 LLM-as-judge（节省 API 费用）
python run_eval.py --top-k 5        # 自定义检索 top-k

# 结果文件
eval_raw_results.json   # 原始评估数据（供进一步分析）
report.md               # 格式化评估报告（含 Markdown 表格）
```

> `eval_answer_quality` 和报告的自动结论需要有效的 `ANTHROPIC_API_KEY`；
> 路由准确率和检索 Recall@k 只用本地 Ollama，无需 API Key。

### 三个评估指标的关系

```
用户问题
   │
   ▼ 模块A：意图路由准确率（eval_routing）
   │   ← 路由错了，后续必然跑偏。应优先保证此模块达标。
   │
   ├── 知识型 → 模块B：RAG 检索 Recall@k（eval_retrieval）
   │             ← 路由对了但检索漏掉，LLM 无法给出正确答案。
   │
   └── (检索命中后) → 模块C：回答质量（eval_answer_quality）
                       ← 前两步都对，才有意义评估生成质量。
```

**三个模块应分开看，不要混为一谈：**
- 路由误判 → 优先改 `router.py` 的 prompt
- 检索未命中 → 考虑缩小 `CHUNK_SIZE` 或换更强的 `EMBEDDING_MODE`
- 生成质量低 → 调整 `generate.py` 的 prompt 模板（引用格式要求、防幻觉约束）

### LLM-as-judge 原理、优点与局限性

**原理**：把"用户问题 + 标准答案摘要 + AI 实际回答"打包发给一个强 LLM（如 Claude），让它按维度打分（准确性 / 引用 / 幻觉），返回结构化 JSON 分数和理由。相比 BLEU/ROUGE 等字符串匹配指标，judge 模型能理解语义、判断引用是否到位、识别捏造内容。这是 RAGAS、G-Eval 等主流 RAG 评估框架的核心思路。

**优点**：
- 语义理解：能判断"意思对不对"，不依赖字符串相似度
- 细粒度：可对准确性、引用、幻觉分别评分
- 低成本：一次调用评估一条，比人工审核快 100×

**局限性**：
- judge 自身也可能出错，尤其是微妙的事实判断
- 打分可能受 prompt 措辞影响，存在偏差（positional bias 等）
- **建议：对得分 ≤ 2 的条目人工抽查**，核实 judge 判断是否合理

### 如何用这份报告排查问题

| 报告中发现 | 应该去改什么 |
|-----------|-------------|
| 路由误判某类问题（尤其模糊题） | `router.py` 中对应意图的 `description` 字段，或 `_JSON_PARSE_SYSTEM` 的 prompt |
| 检索 Recall 低，且同一文档反复未命中 | `CHUNK_SIZE` 调小（当前 400），让相关条款集中在一块而不被稀释；或换 `EMBEDDING_MODE=openai` 用更高维向量 |
| 检索命中了正确文档但回答仍然说"暂无规定" | chunk 切分位置不对，关键条款被切在两块边界；查看 `chroma_db` 里该文档的 chunk 内容 |
| 回答引用分低（< 3） | `generate.py` 的 `build_rag_prompt` 中强化引用要求，如"每个事实点必须标注【来源：文档名第X条】" |
| 回答幻觉分低（< 3） | `build_rag_prompt` 开头加"严禁使用参考材料之外的信息，不确定的内容请直接说没有相关规定" |
