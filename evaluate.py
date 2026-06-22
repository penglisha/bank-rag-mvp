"""
evaluate.py — 自动化评估模块

分三个独立函数，对 RAG + Agent 系统的不同环节打分：

  eval_retrieval      — RAG 检索准确率（Recall@k）
  eval_routing        — 意图路由准确率（json_parse vs tool_calling）
  eval_answer_quality — 回答质量（LLM-as-judge，调用 Claude API）

设计原则：
  - 每个函数独立可运行，互不依赖
  - 返回值统一为结构化 dict，方便 run_eval.py 汇总
  - 出现错误时记录到结果中而非直接崩溃，确保其他条目继续跑
"""

import sys
import os
import json
import time
from typing import Optional

sys.path.insert(0, "src")


# ══════════════════════════════════════════════════════════════
# 模块 B：RAG 检索准确率评估（eval_retrieval）
# ══════════════════════════════════════════════════════════════

def eval_retrieval(dataset: list, embed_fn, db_path: str, top_k: int = 3) -> dict:
    """
    【评估什么】
    对数据集中每条"知识型"问题，跑一次向量检索，
    看 top-k 结果里是否出现了 expected_source_doc 对应的文档。

    【为什么重要】
    RAG 系统的质量上限由检索决定——检索没找到正确文档，
    LLM 再强也无法给出有依据的回答。检索准确率是
    "回答质量"的先决条件，必须单独评估。

    【指标：Recall@k】
    Recall@k = 正确文档出现在 top-k 结果中的问题数 / 知识型问题总数
    k=1：最严格，只看第一名是否命中；
    k=3：宽松，只要前三有一个命中即算成功（本项目默认）

    【参数】
    dataset  : eval_dataset.json 加载后的列表
    embed_fn : embedding.py 返回的向量化函数（已初始化）
    db_path  : Chroma 向量库路径
    top_k    : 检索返回的文档块数量

    【返回结构】
    {
      "top_k"       : int,
      "total"       : int,       # 知识型问题总数
      "hits"        : int,       # 命中数
      "recall_at_k" : float,     # 命中率 [0, 1]
      "details"     : [          # 每条的详细结果
        {
          "id"           : str,
          "question"     : str,
          "expected_doc" : str,
          "hit"          : bool,
          "retrieved_docs": [str],   # 检索到的文档名列表（含分数）
          "top1_score"   : float,    # 第一名的相似度分数
        }
      ],
      "misses"      : [...],     # 未命中条目的列表（方便定位问题）
    }
    """
    from vectorstore import search_vectorstore

    # 只评估知识型问题（查询型/操作型走工具，不走 RAG）
    knowledge_items = [d for d in dataset if d["expected_intent"] == "知识型"]

    hits = 0
    details = []
    misses = []

    print(f"\n{'─' * 56}")
    print(f"  📊 模块B：RAG 检索准确率评估（Recall@{top_k}）")
    print(f"  知识型问题共 {len(knowledge_items)} 条")
    print(f"{'─' * 56}")

    for item in knowledge_items:
        qid       = item["id"]
        question  = item["question"]
        expected  = item["expected_source_doc"]

        # 运行向量检索
        try:
            results = search_vectorstore(question, embed_fn, db_path, top_k=top_k)
        except Exception as e:
            print(f"  [{qid}] ❌ 检索异常：{e}")
            details.append({
                "id": qid, "question": question,
                "expected_doc": expected, "hit": False,
                "retrieved_docs": [], "top1_score": 0.0,
                "error": str(e),
            })
            misses.append({"id": qid, "question": question, "reason": f"检索异常：{e}"})
            continue

        # 检索结果：[(text, metadata, score), ...]
        retrieved_docs = []
        for text, meta, score in results:
            doc_name = meta.get("doc_name", meta.get("filename", "?"))
            retrieved_docs.append({"doc_name": doc_name, "score": round(score, 4)})

        retrieved_names = [r["doc_name"] for r in retrieved_docs]
        top1_score = retrieved_docs[0]["score"] if retrieved_docs else 0.0

        # 判断是否命中：expected_source_doc 出现在 top-k 任意一个结果中
        hit = expected in retrieved_names
        if hit:
            hits += 1
            mark = "✅"
        else:
            mark = "❌"
            misses.append({
                "id": qid,
                "question": question,
                "expected_doc": expected,
                "retrieved_docs": retrieved_docs,
                "reason": f"期望命中《{expected}》，实际检索到：{retrieved_names}",
            })

        # 打印每条结果
        short_q = question[:40] + "…" if len(question) > 40 else question
        print(f"  [{qid}] {mark} {short_q}")
        print(f"         期望文档：《{expected}》")
        for r in retrieved_docs:
            hit_flag = " ← ✅命中" if r["doc_name"] == expected else ""
            print(f"         检索到  ：《{r['doc_name']}》 相似度 {r['score']:.4f}{hit_flag}")

        details.append({
            "id": qid,
            "question": question,
            "expected_doc": expected,
            "hit": hit,
            "retrieved_docs": retrieved_docs,
            "top1_score": top1_score,
        })

    total = len(knowledge_items)
    recall = hits / total if total > 0 else 0.0

    print(f"\n  结果：{hits}/{total} 命中  Recall@{top_k} = {recall:.1%}")
    if misses:
        print(f"  未命中 {len(misses)} 条，见 misses 字段")

    return {
        "top_k":        top_k,
        "total":        total,
        "hits":         hits,
        "recall_at_k":  recall,
        "details":      details,
        "misses":       misses,
    }


# ══════════════════════════════════════════════════════════════
# 模块 A：意图路由准确率评估（eval_routing）
# ══════════════════════════════════════════════════════════════

def eval_routing(dataset: list, modes: list = None) -> dict:
    """
    【评估什么】
    对数据集每条问题（含知识/查询/操作/模糊），分别用两种路由方式
    (json_parse / tool_calling) 判断意图，与 expected_intent 对比。

    【为什么重要】
    意图路由是 Agent 的第一道关卡。路由判断错了，后续的工具调用和
    RAG 检索就全部走错分支，最终答案必然跑偏。
    评估路由准确率能让我们知道：哪类问题容易被误判、哪种路由方式
    在模糊问题上更稳——从而决定应该优化 prompt 还是换更强的路由模型。

    【模糊题处理】
    expected_intent 为 null 的条目是模糊题，不计入准确率分母，
    但仍然跑一遍记录判断结果，方便分析两种方式的"模糊题策略"差异。

    【返回结构】
    {
      "modes": { "json_parse": {...}, "tool_calling": {...} },
      "per_item": [ { "id", "question", "expected", "results": {mode: {...}} } ]
    }
    """
    import importlib

    if modes is None:
        modes = ["json_parse", "tool_calling"]

    print(f"\n{'─' * 56}")
    print(f"  📊 模块A：意图路由准确率评估")
    print(f"  路由方式：{' / '.join(modes)}")
    print(f"{'─' * 56}")

    # 有效题（有明确预期答案的）和模糊题分开统计
    scorable = [d for d in dataset if d["expected_intent"] is not None]
    ambiguous = [d for d in dataset if d["expected_intent"] is None]

    # 结果容器
    mode_stats = {m: {"correct": 0, "total": len(scorable), "errors": []} for m in modes}
    per_item = []

    for item in dataset:
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_intent"]
        is_ambiguous = expected is None

        row = {"id": qid, "question": question, "expected": expected, "results": {}}

        short_q = question[:42] + "…" if len(question) > 42 else question
        prefix = "[模糊]" if is_ambiguous else f"[{expected}]"
        print(f"\n  {qid} {prefix} {short_q}")

        for mode in modes:
            # 切换路由模式：修改环境变量后重新加载 router 模块
            os.environ["ROUTING_MODE"] = mode
            import router as rt
            importlib.reload(rt)

            t0 = time.time()
            try:
                result = rt.route(question)
            except Exception as e:
                got_intent = "ERROR"
                reason     = str(e)
                fallback   = False
                elapsed_ms = int((time.time() - t0) * 1000)
                print(f"    {mode:<14}: ❌ 异常 — {e}")
            else:
                got_intent = result["intent"]
                reason     = result["reason"]
                fallback   = result["fallback"]
                elapsed_ms = int((time.time() - t0) * 1000)

            # 计分（模糊题不计分）
            if not is_ambiguous:
                correct = (got_intent == expected)
                if correct:
                    mode_stats[mode]["correct"] += 1
                    mark = "✅"
                else:
                    mark = "❌"
                    mode_stats[mode]["errors"].append({
                        "id": qid, "question": question,
                        "expected": expected, "got": got_intent,
                        "reason": reason, "fallback": fallback,
                    })
            else:
                correct = None
                mark = "—"

            fb_note = " ↩" if fallback else ""
            print(f"    {mode:<14}: {got_intent:<5} {mark}{fb_note}  ({elapsed_ms}ms)")
            print(f"    {'':14}  依据: {reason[:52]}")

            row["results"][mode] = {
                "got_intent": got_intent,
                "correct":    correct,
                "reason":     reason,
                "fallback":   fallback,
                "elapsed_ms": elapsed_ms,
            }

        per_item.append(row)

    # 汇总
    modes_summary = {}
    for mode in modes:
        s = mode_stats[mode]
        accuracy = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        modes_summary[mode] = {
            "correct":  s["correct"],
            "total":    s["total"],
            "accuracy": accuracy,
            "errors":   s["errors"],
        }
        print(f"\n  {mode}: {s['correct']}/{s['total']}  准确率 {accuracy:.1%}")
        if s["errors"]:
            print(f"    误判条目：" + ", ".join(e["id"] for e in s["errors"]))

    return {
        "modes":    modes_summary,
        "per_item": per_item,
    }


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 模块 C：回答质量评估（eval_answer_quality，LLM-as-judge）
# ══════════════════════════════════════════════════════════════

# ─── 裁判模型配置 ─────────────────────────────────────────────
# 【为什么裁判模型要和生成模型不同？——Self-preference Bias】
#
# 如果用 qwen2.5 生成回答、又用 qwen2.5 打分，会出现"自我偏好偏差"
# (self-preference bias)：同一个模型倾向于给自己风格的回答打高分，
# 对措辞不同但内容正确的回答则可能打低分。这不是客观评估，
# 而是在测量"和自己说话有多像"。
#
# 解决方式：用不同家族/不同训练目标的模型做裁判。
#   - 生成模型（OLLAMA_MODEL 环境变量，默认）：qwen2.5:7b
#   - 裁判模型（JUDGE_MODEL 常量）         ：gemma4-hermes:latest
#
# gemma4-hermes 基于 Hermes-3 指令微调，擅长遵循结构化输出指令，
# 和 qwen2.5 系列在训练数据和风格上有明显差异，适合做独立裁判。
#
# 如果本地没有 gemma4-hermes，先执行：
#   ollama pull gemma4-hermes
# ──────────────────────────────────────────────────────────────
JUDGE_MODEL      = os.getenv("JUDGE_MODEL", "gemma4-hermes:latest")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 裁判 prompt：要求输出扁平 JSON，避免嵌套结构（嵌套更容易让小模型犯格式错误）
_JUDGE_PROMPT = """\
你是一位严格、客观的评估专家。请评估下方 AI 回答的质量。

【用户问题】
{question}

【标准答案摘要（仅供参考，不需要逐字匹配）】
{expected_summary}

【AI 实际回答】
{actual_answer}

【评估说明】
- accuracy_score（整数 1-5）：回答内容与标准答案是否一致、关键信息是否覆盖
  1=严重偏差或缺失关键信息，3=部分正确，5=准确且完整
- has_citation（布尔值）：回答中是否出现了具体的文档名称或条款编号（如"第6.1款"）
  true=有引用，false=没有任何来源标注
- hallucination_detected（布尔值）：回答中是否存在材料之外的编造内容
  true=存在编造，false=完全基于给定材料
- reason（字符串）：一句话综合说明打分理由

只输出以下 JSON，不要有其他任何文字：
{{"accuracy_score": 1, "has_citation": true, "hallucination_detected": false, "reason": "理由"}}
"""


def _call_ollama_judge(prompt: str, judge_model: str) -> str:
    """
    调用 Ollama 本地模型，使用 format:"json" 强制输出合法 JSON。
    使用 httpx 绕过系统代理（trust_env=False），避免 localhost 被代理拦截。
    返回模型的原始输出字符串。
    """
    import httpx
    client = httpx.Client(trust_env=False)
    resp = client.post(
        f"{_OLLAMA_BASE_URL}/api/chat",
        json={
            "model": judge_model,
            "format": "json",   # Ollama 结构化输出：强制合法 JSON，避免多余文字
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},  # 温度0让打分更确定性，减少随机波动
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def eval_answer_quality(
    dataset: list,
    embed_fn,
    db_path: str,
    judge_model: str = JUDGE_MODEL,
) -> dict:
    """
    【评估什么】
    对数据集中每条"知识型"问题，先跑完整 RAG 流程得到实际回答，
    再用本地 Ollama 模型作为"裁判"（LLM-as-judge），评估三个维度：
      - accuracy_score  : 准确性得分（1-5）
      - has_citation    : 是否标注了引用来源（True/False）
      - hallucination_detected : 是否存在编造内容（True/False）

    【为什么用 LLM-as-judge 而不是传统指标】
    BLEU/ROUGE 等字符串匹配方法无法判断语义是否正确，
    也无法检测"答案对但没有引用"或"回答流畅但内容是编造的"这类问题。
    LLM-as-judge 能做语义理解，是 RAG 评估的业界主流方案
    （G-Eval、RAGAS 等框架的核心思路）。

    【本地模型 vs 云端 API 的真实差异】
    使用本地小模型（7B~10B）做裁判时，存在以下已知局限，不要回避：
      1. 评分一致性变差：同一道题重复跑两次可能给出不同分数
      2. 边界分辨力弱：对 3 分和 4 分之间的模糊情况难以区分
      3. 语言理解偏差：中文混合专业术语时，小模型可能误读内容
      4. 格式偶发失控：即使有 format:json，极少数情况下仍可能输出无效 JSON

    因此：跑完之后建议人工抽查 3-5 条打分结果，尤其是极端分（1分或5分），
    核实裁判判断是否符合你自己的理解，不要完全依赖自动化数字。

    【参数】
    judge_model : 裁判模型（默认 JUDGE_MODEL 常量，与生成模型不同以避免自我偏好偏差）
    """
    from vectorstore import search_vectorstore
    from generate import generate_answer

    knowledge_items = [d for d in dataset if d["expected_intent"] == "知识型"]

    print(f"\n{'─' * 56}")
    print(f"  📊 模块C：回答质量评估（LLM-as-judge，本地 Ollama）")
    print(f"  裁判模型：{judge_model}")
    print(f"  生成模型：{os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')}（两者不同，避免 self-preference bias）")
    print(f"  知识型问题：{len(knowledge_items)} 条")
    print(f"{'─' * 56}")

    details      = []
    acc_total    = 0
    citation_hits = 0
    halluc_hits  = 0
    scored_count = 0

    for item in knowledge_items:
        qid              = item["id"]
        question         = item["question"]
        expected_summary = item["expected_answer_summary"]

        short_q = question[:42] + "…" if len(question) > 42 else question
        print(f"\n  [{qid}] {short_q}")

        # ── Step 1：RAG 检索 + 生成回答 ─────────────────────────
        try:
            retrieved    = search_vectorstore(question, embed_fn, db_path, top_k=3)
            actual_answer = generate_answer(question, retrieved)
        except Exception as e:
            print(f"    ❌ RAG 生成失败：{e}")
            details.append({"id": qid, "question": question, "rag_error": str(e)})
            continue

        print(f"    生成回答（前80字）：{actual_answer[:80].replace(chr(10), ' ')}…")

        # ── Step 2：调用裁判模型打分，失败最多重试一次 ───────────
        judge_prompt = _JUDGE_PROMPT.format(
            question=question,
            expected_summary=expected_summary,
            actual_answer=actual_answer,
        )

        raw_scores = None
        for attempt in range(2):   # 最多尝试2次
            try:
                raw_scores = _call_ollama_judge(judge_prompt, judge_model)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"    ⚠️  judge 第1次调用失败（{e}），重试…")
                else:
                    print(f"    ❌ judge 调用失败（重试后仍失败）：{e}")
                    details.append({
                        "id": qid, "question": question,
                        "actual_answer": actual_answer,
                        "judge_error": str(e),
                    })

        if raw_scores is None:
            continue

        # ── Step 3：解析 JSON，失败记录并跳过（不崩溃整个流程）──
        scores = _parse_judge_json(raw_scores)
        if scores is None:
            print(f"    ⚠️  JSON 解析失败，跳过（原始：{raw_scores[:80]}）")
            details.append({
                "id": qid, "question": question,
                "actual_answer": actual_answer,
                "raw_scores": raw_scores,
                "parse_error": True,
            })
            continue

        # ── Step 4：提取各字段，容错处理类型不规范的情况 ─────────
        acc_score  = _safe_int(scores.get("accuracy_score"), 1, 5)
        has_cit    = _safe_bool(scores.get("has_citation"))
        halluc     = _safe_bool(scores.get("hallucination_detected"))
        reason     = str(scores.get("reason", "（无理由）"))[:60]

        acc_total     += acc_score
        citation_hits += int(has_cit)
        halluc_hits   += int(halluc)
        scored_count  += 1

        # 打印本条结果
        cit_str   = "✅有引用" if has_cit  else "❌无引用"
        halluc_str = "⚠️有幻觉" if halluc  else "✅无幻觉"
        print(f"    准确:{acc_score}/5  {cit_str}  {halluc_str}")
        print(f"    理由：{reason}")

        details.append({
            "id":                    qid,
            "question":              question,
            "expected_summary":      expected_summary,
            "actual_answer":         actual_answer,
            "scores": {
                "accuracy_score":        acc_score,
                "has_citation":          has_cit,
                "hallucination_detected": halluc,
                "reason":                reason,
            },
        })

    # ── 汇总统计 ──────────────────────────────────────────────
    avg_scores = {}
    if scored_count > 0:
        avg_scores = {
            "accuracy":          round(acc_total / scored_count, 2),
            "citation_rate":     round(citation_hits / scored_count, 2),   # 有引用的比例
            "hallucination_rate": round(halluc_hits / scored_count, 2),    # 检出幻觉的比例（越低越好）
        }

    print(f"\n  评分完成 {scored_count}/{len(knowledge_items)} 条")
    if avg_scores:
        print(f"  平均准确分：{avg_scores['accuracy']}/5")
        print(f"  引用率：{avg_scores['citation_rate']:.0%}（有引用的比例）")
        print(f"  幻觉率：{avg_scores['hallucination_rate']:.0%}（检出幻觉的比例，越低越好）")

    return {
        "judge_model":  judge_model,
        "total":        len(knowledge_items),
        "scored":       scored_count,
        "avg_scores":   avg_scores,
        "details":      details,
    }


def _safe_int(val, lo: int, hi: int) -> int:
    """把不规范的 score 值（字符串/越界数字）强制转为合法整数。"""
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return lo


def _safe_bool(val) -> bool:
    """把不规范的布尔值（字符串'true'/'false'、0/1 等）统一转为 bool。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "是")
    return False


def _parse_judge_json(text: str) -> Optional[dict]:
    """从 judge 输出中提取 JSON，支持多种格式容错。"""
    import re
    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取 markdown 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 提取最外层花括号
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
