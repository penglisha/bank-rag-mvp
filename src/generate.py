"""
generate.py — 回答生成模块（RAG 第四步，最终步骤）

【RAG 中 LLM 的作用】
检索步骤找到的是"原始文本片段"，可能存在以下问题：
  - 多个片段各自描述问题的不同侧面，需要综合
  - 片段语言可能生硬，不适合直接呈现给用户
  - 用户问题与片段措辞不同，需要"翻译"对应

LLM 在这里扮演的角色是：理解检索到的材料 + 理解用户问题 → 生成流畅、准确、有引用的回答。
这就是 RAG 的精髓：检索负责"找到事实"，LLM 负责"理解和表达"。

【为什么要在 Prompt 中限制"只根据材料回答"？】
如果不限制，LLM 会用自己的训练知识回答，可能出现"幻觉"（看似合理但实际错误的信息）。
在合规场景下，错误信息可能导致严重后果，因此必须严格限定信息来源。
"""

import os
from typing import List, Tuple
import anthropic


def build_rag_prompt(query: str, retrieved_chunks: List[Tuple[str, dict, float]]) -> str:
    """
    构建 RAG 的完整 Prompt。

    【好的 RAG Prompt 的设计要点】
    1. "只基于以下材料回答" — 防止模型引入材料之外的知识（防幻觉）
    2. 要求标注来源引用 — 提高回答可信度，方便人工核查原文
    3. "材料不足时明确说明" — 让模型拒绝而不是编造
    4. 用视觉分隔符区分"材料区"和"问题区" — 结构清晰，模型更好理解
    5. 在材料中附上相关度分数 — 帮助模型判断哪些材料更可信

    参数：
        query            : 用户输入的原始问题
        retrieved_chunks : vectorstore.search 返回的 [(文本, 元数据, 相似度), ...]
    """
    # 格式化参考材料，带编号方便模型引用
    context_parts = []
    for i, (text, meta, score) in enumerate(retrieved_chunks, 1):
        doc_name = meta.get('doc_name', meta.get('filename', '未知文档'))
        context_parts.append(
            f"【参考材料 {i}】\n"
            f"来源：《{doc_name}》（检索相关度：{score:.1%}）\n\n"
            f"{text}"
        )

    context = "\n\n" + ("─" * 50 + "\n\n").join(context_parts)

    prompt = f"""你是一位专业的银行合规知识助手，负责解答银行产品和合规相关问题。

【重要规则】
1. 严格只基于下方"参考材料"中的内容回答，不得引用材料之外的信息
2. 回答中必须用【来源：《文档名》第X条/第X.X款】格式标注具体引用位置
3. 如有多条相关规定，请逐一说明并分别标注来源
4. 如参考材料中没有足够信息，请明确回答"根据现有材料暂无相关规定"
5. 回答要简洁专业，直接回答问题，无需重复问题本身

━━━━━━━━━ 参考材料 ━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户问题：{query}

请基于以上参考材料，给出准确、有引用来源的回答："""

    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[str, dict, float]],
    model: str = None
) -> str:
    """
    调用 Anthropic Claude API，基于检索结果生成最终回答。

    【模型选择说明】
    默认使用 claude-haiku-4-5：
    - 速度最快（约1-2秒），适合问答类交互场景
    - 成本最低（约为 Opus 的1/20），适合学习时频繁实验
    - 对指令遵循能力强，能按要求格式输出引用
    可通过 .env 中的 ANTHROPIC_MODEL 切换至更强的模型。

    参数：
        query            : 用户问题
        retrieved_chunks : 检索到的相关文本块
        model            : 覆盖环境变量指定的模型（可选）

    返回：
        带引用来源的完整回答字符串
    """
    # 根据 LLM_BACKEND 环境变量选择后端（默认 anthropic）
    # - anthropic：调用 Anthropic Claude API，需要 ANTHROPIC_API_KEY
    # - ollama：调用本地 Ollama，完全免费，需要先安装 Ollama 并拉取模型
    backend = os.getenv('LLM_BACKEND', 'anthropic').lower()

    prompt = build_rag_prompt(query, retrieved_chunks)

    if backend == 'ollama':
        return _generate_ollama(prompt)
    else:
        return _generate_anthropic(prompt, model)


def _generate_anthropic(prompt: str, model: str = None) -> str:
    """使用 Anthropic Claude API 生成回答。"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError(
            "未找到 ANTHROPIC_API_KEY。\n"
            "请在 .env 中填入 ANTHROPIC_API_KEY，\n"
            "或设置 LLM_BACKEND=ollama 改用本地免费模型。"
        )

    if model is None:
        model = os.getenv('ANTHROPIC_MODEL', 'claude-haiku-4-5')

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def _generate_ollama(prompt: str) -> str:
    """
    使用本地 Ollama 生成回答（完全免费，无需 API Key）。

    前置条件：
      1. 安装 Ollama：https://ollama.com/download
      2. 拉取中文模型：ollama pull qwen2.5:7b
      3. 在 .env 中设置：LLM_BACKEND=ollama

    Ollama 提供 OpenAI 兼容接口（base_url=http://localhost:11434/v1），
    用 openai 包即可调用，无需额外安装。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Ollama 后端需要 openai 包作为客户端：pip install openai\n"
            "（仅用于调用本地 Ollama，不会向 OpenAI 发送任何请求）"
        )

    ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')

    try:
        import httpx
    except ImportError:
        raise ImportError("请安装 httpx：pip install httpx")

    # trust_env=False：禁止 httpx 读取系统代理环境变量（HTTP_PROXY / HTTPS_PROXY）。
    # 不加这行时，若系统配置了代理，发往 localhost 的请求也会被转发给代理，
    # 代理无法处理本地服务请求，导致 "Connection error" / "Server disconnected"。
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=httpx.Client(trust_env=False)
    )

    response = client.chat.completions.create(
        model=ollama_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
