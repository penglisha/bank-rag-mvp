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
│   └── main.py                 # 主入口、CLI 交互、打印中间过程
├── chroma_db/                  # 向量库（自动生成，删除后 --rebuild 重建）
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

1. **改变 CHUNK_SIZE**：分别设置 200 / 400 / 800，问同一个问题，对比检索片段的精准度
2. **改变 TOP_K**：设置 1 vs 5，观察回答完整性的差异
3. **切换 LLM 后端**：同一问题分别用 Ollama 和 Claude 回答，对比质量
4. **切换 EMBEDDING_MODE**：对比 local 和 openai 的相似度分数差异（需 OpenAI Key + `--rebuild`）
5. **添加新文档**：在 `docs/` 放入新 .txt 文件，`python src/main.py --rebuild` 重建

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

**Q: 想用其他 Ollama 模型（如 llama3、gemma）**
A: `ollama pull <model_name>` 拉取后，在 `.env` 中设置 `OLLAMA_MODEL=<model_name>` 即可。
