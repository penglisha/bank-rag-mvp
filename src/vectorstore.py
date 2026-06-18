"""
vectorstore.py — 向量数据库模块（RAG 第三步）

【为什么需要向量数据库？】
有了文本向量后，还需要一个地方来高效存储和检索。
普通数据库只能做精确匹配，向量数据库能做"最近邻搜索"：
  找到与查询向量"方向最接近"的那些存储向量 → 对应语义最相关的文本块

本模块使用 Chroma：
- 纯 Python 实现，无需单独启动服务，数据存本地文件夹
- 支持持久化：重启程序后向量库数据不丢失
- 支持元数据过滤（如按文档名过滤）

【相似度计算：余弦相似度】
两个向量的夹角越小（方向越接近），余弦值越大（越接近1），语义越相近。
公式：cos(θ) = (A·B) / (|A|·|B|)
Chroma 存的是"余弦距离" = 1 - 余弦相似度，越小越相关。
我们在返回时转换为"相似度"，直觉上更好理解（越大越相关）。

【Embedding 由外部负责，Chroma 只做存储和检索】
本模块不调用任何 embedding 模型。向量由 embedding.py 的 embed_fn 计算好后
通过参数传入，写入时用 embeddings= 参数，查询时用 query_embeddings= 参数。
创建集合时显式传 embedding_function=None，阻止 Chroma 加载它自己的默认模型
（DefaultEmbeddingFunction = all-MiniLM-L6-v2），避免不必要的模型下载和内存占用。
"""

import os
from typing import List, Tuple, Callable

import chromadb
from chromadb.config import Settings


def _get_client(db_path: str) -> chromadb.PersistentClient:
    """创建持久化 Chroma 客户端（数据存在 db_path 目录）。"""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)  # 关闭匿名数据收集
    )


def get_or_create_collection(db_path: str, collection_name: str = "bank_docs"):
    """
    获取已有集合，若不存在则创建新集合。

    collection（集合）相当于关系数据库中的"表"，
    一个 Chroma 数据库可以有多个集合，按业务场景分开存储。

    hnsw:space = "cosine" 指定使用余弦距离作为向量间的距离度量方式。

    embedding_function=None：告诉 Chroma "我们自己管理向量化，不需要你的内置模型"。
    不传这个参数时 Chroma 会默认加载 DefaultEmbeddingFunction（all-MiniLM-L6-v2），
    虽然我们不会调用它，但它仍会被下载和加载，浪费资源并可能引起混淆。
    """
    client = _get_client(db_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,   # ← 关键：禁用 Chroma 内置模型，由 embed_fn 负责
        metadata={"hnsw:space": "cosine"}
    )


def build_vectorstore(
    chunks: List[dict],
    embed_fn: Callable,
    db_path: str,
    collection_name: str = "bank_docs",
    batch_size: int = 50
) -> None:
    """
    将文本块向量化并批量写入 Chroma。

    流程：
      文本块列表 → 批量调用 embed_fn 向量化 → 写入 Chroma（文本+向量+元数据）

    参数：
        chunks       : chunking.py 输出的文本块列表
        embed_fn     : embedding.py 返回的向量化函数
        db_path      : 向量库存储目录路径
        collection_name : 集合名称
        batch_size   : 每批处理的块数（避免一次发送太多导致内存溢出或 API 限速）

    【幂等性设计】
    如果集合中已有数据，跳过构建（不重复写入）。
    需要重建时，请先调用 clear_vectorstore()。
    """
    collection = get_or_create_collection(db_path, collection_name)

    existing_count = collection.count()
    if existing_count > 0:
        print(f"    ✅ 向量库已存在，共 {existing_count} 个文本块，跳过重建")
        print(f"    （如需重建，运行时加 --rebuild 参数）")
        return

    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    # Chroma 要求每条记录有唯一 ID
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"    开始向量化并写入，共 {len(texts)} 个文本块（batch_size={batch_size}）...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # ★ 核心操作：文本 → 向量（调用 OpenAI API 或本地模型）
        embeddings = embed_fn(batch_texts)

        # 将向量、原文、元数据一起存入 Chroma
        # 同时存原文是为了检索后直接返回文本，无需再查一次数据库
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids
        )

        done = min(i + batch_size, len(texts))
        print(f"    进度：{done}/{len(texts)} 块已写入")

    print(f"    ✅ 向量库构建完成，共写入 {len(texts)} 个文本块")


def search_vectorstore(
    query: str,
    embed_fn: Callable,
    db_path: str,
    top_k: int = 3,
    collection_name: str = "bank_docs"
) -> List[Tuple[str, dict, float]]:
    """
    根据用户问题，检索最相关的 top_k 个文本块。

    流程：
      用户问题 → embed_fn 向量化 → 在 Chroma 中最近邻搜索 → 返回 top_k 结果

    返回格式：
      [(文本内容, 元数据字典, 相似度分数), ...]
      相似度分数范围 [0, 1]，越接近 1 越相关
    """
    collection = get_or_create_collection(db_path, collection_name)

    # 将用户问题也转成向量（与存储的文本块向量在同一向量空间才能比较）
    query_embedding = embed_fn([query])[0]

    # 在向量库中找最近的 top_k 个向量
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # 防止 top_k > 库中总量
        include=['documents', 'metadatas', 'distances']
    )

    # 整理结果，将余弦距离转换为相似度
    # Chroma 返回的 distance = 1 - cosine_similarity（余弦距离）
    # 所以 similarity = 1 - distance
    output = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = 1.0 - dist
        output.append((doc, meta, similarity))

    return output


def clear_vectorstore(db_path: str, collection_name: str = "bank_docs") -> None:
    """
    删除指定集合（用于重建向量库）。
    调用后再次 build_vectorstore 即可从头重建。
    """
    try:
        client = _get_client(db_path)
        client.delete_collection(collection_name)
        print(f"    🗑️  已清空向量库集合：{collection_name}")
    except Exception:
        pass  # 集合不存在时静默忽略
