"""
chunking.py — 文档切块模块（RAG 第一步）

【为什么要切块？】
RAG 的核心思路是"检索再生成"。如果把整篇文档丢给模型，有两个问题：
1. 模型 context window 有限，无法塞入太多文档
2. 向量化整篇文档后，向量只能表达全文的"平均语义"，无法精确匹配局部问题

切块的本质是：把大文档拆成聚焦单一主题的小片段，
使得每个片段的向量能精确代表该片段的语义，检索时才能"按图索骥"。

【切块策略说明】
本模块采用"段落优先 + 长度兜底"的策略：
- 优先按段落（双换行符）切分，保持语义完整性
- 相邻块之间保留 overlap（重叠），避免重要信息被切断在边界处
- 单个段落超过 chunk_size 时，强制按字符切分
"""

import os
from typing import List, Dict


def load_documents(docs_dir: str) -> List[Dict]:
    """
    加载指定目录下所有 .txt 文件。
    返回格式：[{'filename': '...', 'content': '...', 'filepath': '...'}, ...]
    """
    documents = []
    for filename in sorted(os.listdir(docs_dir)):  # sorted 保证加载顺序一致
        if filename.endswith('.txt'):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({
                'filename': filename,
                'content': content,
                'filepath': filepath
            })
    return documents


def chunk_document(doc: Dict, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    将单篇文档切分为文本块列表。

    参数：
        doc        : load_documents 返回的单个文档字典
        chunk_size : 每块的最大字符数（默认 400）
        overlap    : 相邻块之间的重叠字符数（默认 50）
                     重叠的作用：如果关键句子恰好在切分边界，
                     overlap 能确保这句话同时出现在前后两个块中，不会"丢失"

    返回：
        [{'text': '...', 'metadata': {'filename':..., 'chunk_id':..., 'doc_name':...}}, ...]
    """
    chunks = []
    content = doc['content']
    filename = doc['filename']
    doc_name = filename.replace('.txt', '')

    # 按两个换行符分段（对应文档中空行分隔的段落）
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        # 若当前段落与已有内容合并后不超过限制，则追加合并
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        else:
            # 先保存当前块（若非空）
            if current_chunk:
                chunks.append(_make_chunk(current_chunk, filename, doc_name, chunk_id))
                chunk_id += 1
                # 用上一块末尾的 overlap 个字符作为新块的"热身"起点
                tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (tail + "\n\n" + para).strip()
            else:
                current_chunk = para

            # 单个段落本身超过 chunk_size 时，强制切分
            while len(current_chunk) > chunk_size:
                chunks.append(_make_chunk(current_chunk[:chunk_size], filename, doc_name, chunk_id))
                chunk_id += 1
                current_chunk = current_chunk[chunk_size - overlap:]

    # 将最后一个块入列
    if current_chunk.strip():
        chunks.append(_make_chunk(current_chunk.strip(), filename, doc_name, chunk_id))

    return chunks


def _make_chunk(text: str, filename: str, doc_name: str, chunk_id: int) -> Dict:
    """构造单个 chunk 字典（内部辅助函数）。"""
    return {
        'text': text,
        'metadata': {
            'filename': filename,
            'doc_name': doc_name,
            'chunk_id': chunk_id,
        }
    }


def chunk_all_documents(docs_dir: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    加载 docs_dir 目录下所有文档并完成切块。
    这是对外暴露的主函数，main.py 直接调用此函数。

    返回：所有文档切块结果合并后的列表
    """
    documents = load_documents(docs_dir)
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        print(f"    📄 {doc['filename']}: 切分为 {len(chunks)} 个文本块")

    return all_chunks
