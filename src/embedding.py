"""
embedding.py — 文本向量化模块（RAG 第二步）

【为什么要向量化？】
人类能直接理解"逾期利息"和"罚息"是相关的，但计算机不能直接比较字符串语义。
Embedding（向量化）的作用是把文本映射到高维空间中的一个数字向量，
语义相近的文本，其向量在空间中的"方向"也相近（余弦相似度高）。

例如：
  "信用卡逾期多少天会影响征信？"
  "逾期超过90天会上报征信系统"
这两段文字的向量方向会很接近，即使没有共同的关键词。

【两种模式说明】
- local（默认）：使用 sentence-transformers 本地运行，完全免费
  模型：paraphrase-multilingual-MiniLM-L12-v2
  支持中文，向量维度384，首次运行需下载约400MB模型文件

- openai：使用 OpenAI text-embedding-3-small API
  向量维度1536，中文理解质量更高，每次调用需付费（约$0.02/百万tokens）

通过 .env 文件中的 EMBEDDING_MODE 切换。
"""

import os
from typing import List, Callable


def get_embedding_function() -> Callable[[List[str]], List[List[float]]]:
    """
    根据 EMBEDDING_MODE 环境变量返回对应的 embedding 函数。

    返回的函数签名：fn(texts: List[str]) -> List[List[float]]
    输入：文本列表
    输出：对应的向量列表（每个向量是一个浮点数列表）
    """
    mode = os.getenv('EMBEDDING_MODE', 'local').lower()

    if mode == 'openai':
        return _build_openai_embed_fn()
    else:
        return _build_local_embed_fn()


def _build_openai_embed_fn() -> Callable:
    """
    构建 OpenAI embedding 函数。

    使用 text-embedding-3-small：
    - 输出向量维度：1536
    - 支持语言：多语言（中文效果好）
    - 优点：质量高，API 调用简单
    - 缺点：需要 OPENAI_API_KEY，每次调用有成本
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先安装 openai 包：pip install openai")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("使用 OpenAI embedding 需要在 .env 中设置 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """
        批量向量化文本。
        OpenAI API 支持一次请求发送多条文本，节省网络往返次数。
        """
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        # response.data 是按输入顺序排列的 Embedding 对象列表
        return [item.embedding for item in response.data]

    print("    🔌 Embedding 模式：OpenAI text-embedding-3-small（维度1536）")
    return embed_texts


def _build_local_embed_fn() -> Callable:
    """
    构建本地 sentence-transformers embedding 函数。

    使用 paraphrase-multilingual-MiniLM-L12-v2：
    - 输出向量维度：384
    - 支持语言：50+ 种语言，包括中文
    - 优点：完全免费，可离线运行，速度还不错
    - 缺点：首次运行需下载约400MB模型，质量略低于大型API模型

    【为什么选这个模型？】
    "Multilingual"表示多语言支持，对中文银行合规文档效果合格。
    "MiniLM"表示轻量级，在普通笔记本上也能快速运行。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("请先安装 sentence-transformers：pip install sentence-transformers")

    model_name = os.getenv(
        'LOCAL_EMBEDDING_MODEL',
        'paraphrase-multilingual-MiniLM-L12-v2'
    )

    print(f"    🤖 Embedding 模式：本地模型 {model_name}（维度384）")
    print("    （首次运行会自动下载模型文件，约400MB，请耐心等待...）")
    model = SentenceTransformer(model_name)
    print("    ✅ 模型加载完成")

    def embed_texts(texts: List[str]) -> List[List[float]]:
        """
        本地批量向量化。
        encode() 返回 numpy ndarray，需转为 Python list 才能存入 Chroma。
        show_progress_bar=False 避免构建向量库时输出大量进度条干扰主流程。
        """
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    return embed_texts
