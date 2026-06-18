"""
main.py — RAG 系统主入口

将 chunking / embedding / vectorstore / generate 四个模块串联起来，
实现完整的银行合规知识助手 CLI 交互。

【完整数据流】
① 文档(.txt)
        ↓ [chunking.py] 按段落切块
② 文本块列表（含元数据）
        ↓ [embedding.py] 批量向量化
③ 文本向量
        ↓ [vectorstore.py] 写入 Chroma 本地向量库
                                        ↑ 首次运行完成后持久化，后续跳过此步骤

用户提问时：
④ 用户问题
        ↓ [embedding.py] 单条向量化
⑤ 查询向量
        ↓ [vectorstore.py] 余弦相似度搜索，返回 top-k
⑥ 相关文本块（含相似度分数）
        ↓ [generate.py] 拼 Prompt + 调用 Claude API
⑦ 带引用来源的最终回答

【命令行用法】
  python src/main.py             # 正常启动
  python src/main.py --rebuild   # 强制重建向量库（修改文档后使用）
"""

import os
import sys
from pathlib import Path

# 加载 .env 配置（必须在所有其他导入之前）
from dotenv import load_dotenv
load_dotenv()

# 将 src 目录加入模块搜索路径，支持直接 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from chunking import chunk_all_documents
from embedding import get_embedding_function
from vectorstore import build_vectorstore, search_vectorstore, clear_vectorstore
from generate import generate_answer


# ──────────────────────────────────────────────────────────
# 配置区（修改此处或对应的 .env 条目做实验）
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # 项目根目录 bank-rag-mvp/
DOCS_DIR = str(BASE_DIR / "docs")         # 原始文档目录
DB_PATH  = str(BASE_DIR / "chroma_db")   # 向量库持久化目录

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 400))
TOP_K      = int(os.getenv('TOP_K', 3))
# ──────────────────────────────────────────────────────────


def initialize_system(rebuild: bool = False):
    """
    初始化 RAG 系统：加载文档 → 切块 → 向量化 → 写入向量库。

    rebuild=True：强制清空向量库后重建（适用于文档内容有更新时）
    rebuild=False（默认）：若向量库已存在则跳过构建，直接复用

    返回：embed_fn（向量化函数），供后续查询复用（避免重复加载模型）
    """
    print("\n" + "═" * 60)
    print("  🏦 银行合规知识助手 — 系统初始化")
    print("═" * 60)

    if rebuild:
        print("\n🗑️  清空旧向量库...")
        clear_vectorstore(DB_PATH)

    # ── 步骤1：文档切块 ─────────────────────────────────────
    print(f"\n📂 [步骤1] 加载并切块文档（chunk_size={CHUNK_SIZE}）")
    chunks = chunk_all_documents(DOCS_DIR, chunk_size=CHUNK_SIZE, overlap=50)
    print(f"  → 共生成 {len(chunks)} 个文本块")

    # ── 步骤2：初始化 Embedding 模型 ────────────────────────
    mode = os.getenv('EMBEDDING_MODE', 'local')
    print(f"\n🔮 [步骤2] 初始化 Embedding 模型（EMBEDDING_MODE={mode}）")
    embed_fn = get_embedding_function()

    # ── 步骤3：构建向量库 ────────────────────────────────────
    print(f"\n💾 [步骤3] 构建向量库（路径：chroma_db/）")
    build_vectorstore(chunks, embed_fn, DB_PATH, batch_size=50)

    print("\n✅ 初始化完成，可以开始提问！\n")
    return embed_fn


def run_query(query: str, embed_fn) -> str:
    """
    执行完整的 RAG 查询，并打印每一步的中间过程。

    打印中间过程的目的：让你直观看到 RAG 在做什么——
    检索到了哪些材料、相似度多少、最终怎么生成回答。
    这比只看最终答案更有助于理解原理。
    """
    print(f"\n{'═' * 60}")
    print(f"  ❓ 用户问题：{query}")
    print(f"{'═' * 60}")

    # ── 步骤4：向量检索 ─────────────────────────────────────
    print(f"\n🔍 [步骤4] 向量检索（top_k={TOP_K}）")
    retrieved = search_vectorstore(query, embed_fn, DB_PATH, top_k=TOP_K)

    # 打印检索结果——这是理解 RAG 效果的关键环节
    print(f"\n  检索到 {len(retrieved)} 个相关文本块：")
    print("  " + "─" * 56)
    for i, (text, meta, score) in enumerate(retrieved, 1):
        doc_name = meta.get('doc_name', '未知')
        # 预览：截取前80个字符，折叠换行符
        preview = text[:80].replace('\n', ' ')
        if len(text) > 80:
            preview += "..."
        print(f"\n  块 {i} │ 来源：《{doc_name}》")
        print(f"       │ 相似度：{score:.4f}  ({score:.1%})")
        print(f"       │ 内容：{preview}")

    # ── 步骤5：LLM 生成回答 ──────────────────────────────────
    backend = os.getenv('LLM_BACKEND', 'anthropic')
    if backend == 'ollama':
        llm_label = f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')})"
    else:
        llm_label = f"Claude ({os.getenv('ANTHROPIC_MODEL', 'claude-haiku-4-5')})"
    print(f"\n🤖 [步骤5] 调用 {llm_label} 生成回答...")

    answer = generate_answer(query, retrieved)
    return answer


def print_examples():
    """打印示例问题，帮助用户快速上手。"""
    print("\n💡 示例问题（直接复制粘贴试试）：")
    examples = [
        "信用卡逾期30天会有什么后果？",
        "申请个人贷款需要满足哪些条件？",
        "理财产品的风险等级是怎么划分的？",
        "征信报告被频繁查询会有什么影响？",
        "客户投诉后银行需要多久回复？",
        "信用卡逾期罚息是怎么计算的？",
        "什么情况下可以申请提前还款？",
    ]
    for q in examples:
        print(f"  • {q}")
    print()


def main():
    """命令行主入口。"""
    rebuild = "--rebuild" in sys.argv

    # 初始化（切块 + 建向量库）
    embed_fn = initialize_system(rebuild=rebuild)

    # 打印欢迎界面
    print("═" * 60)
    print("  💬 对话模式启动")
    print("  输入问题直接回车，输入 quit 退出，输入 rebuild 重建向量库")
    print("═" * 60)
    print_examples()

    # 主交互循环
    while True:
        try:
            user_input = input("请输入问题 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 感谢使用，再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "退出"):
            print("\n👋 感谢使用，再见！")
            break

        if user_input.lower() == "rebuild":
            embed_fn = initialize_system(rebuild=True)
            continue

        # 执行查询
        try:
            answer = run_query(user_input, embed_fn)
        except ValueError as e:
            print(f"\n❌ 配置错误：{e}")
            continue
        except Exception as e:
            print(f"\n❌ 发生错误：{e}")
            continue

        # 输出最终回答
        print("\n" + "─" * 60)
        print("💡 最终回答")
        print("─" * 60)
        print(answer)
        print("─" * 60)


if __name__ == "__main__":
    main()
