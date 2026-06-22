"""
run_eval.py — 评估主流程

依次运行三个评估模块，把结果汇总成 report.md。
自动结论由本地 Ollama 生成（无需任何 API Key）。

运行方式：
  python run_eval.py              # 跑全部三个模块（含 LLM-as-judge）
  python run_eval.py --no-quality # 跳过回答质量评估（节省时间）
  python run_eval.py --top-k 5    # 自定义检索 top-k
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()

from embedding import get_embedding_function
from evaluate import eval_retrieval, eval_routing, eval_answer_quality, JUDGE_MODEL

# ─── 路径配置 ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
DB_PATH      = str(BASE_DIR / "chroma_db")
REPORT_PATH  = BASE_DIR / "report.md"
RAW_PATH     = BASE_DIR / "eval_raw_results.json"
# ──────────────────────────────────────────────────────────────


def load_dataset() -> list:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════

def _routing_table(routing_result: dict) -> str:
    """生成路由准确率对比表（Markdown）。"""
    modes = list(routing_result["modes"].keys())
    lines = [
        "| 题号 | 预期意图 | " + " | ".join(f"{m} 判断 ✓" for m in modes) + " |",
        "|------|---------|" + "|---------|" * len(modes),
    ]
    for row in routing_result["per_item"]:
        expected = row["expected"] or "模糊"
        cells = []
        for m in modes:
            r = row["results"].get(m, {})
            got      = r.get("got_intent", "—")
            correct  = r.get("correct")
            fallback = r.get("fallback", False)
            if correct is None:
                mark = "—"
            elif correct:
                mark = "✅"
            else:
                mark = "❌"
            fb = "↩" if fallback else ""
            cells.append(f"{got} {mark}{fb}")
        lines.append(f"| {row['id']} | {expected} | " + " | ".join(cells) + " |")

    # 汇总行
    summary_cells = []
    for m in modes:
        s = routing_result["modes"][m]
        summary_cells.append(f"**{s['correct']}/{s['total']} ({s['accuracy']:.0%})**")
    lines.append(f"| **合计** | — | " + " | ".join(summary_cells) + " |")
    return "\n".join(lines)


def _retrieval_table(retrieval_result: dict) -> str:
    """生成检索命中明细表（Markdown）。"""
    lines = [
        "| 题号 | 问题（缩略）| 期望文档 | 命中? | Top1相似度 | 检索到的文档 |",
        "|------|-----------|---------|------|-----------|------------|",
    ]
    for d in retrieval_result["details"]:
        hit    = "✅" if d.get("hit") else "❌"
        short  = d["question"][:22] + "…" if len(d["question"]) > 22 else d["question"]
        top1   = f'{d.get("top1_score", 0):.4f}'
        docs   = "、".join(r["doc_name"] if isinstance(r, dict) else r
                           for r in d.get("retrieved_docs", [])[:3])
        lines.append(f"| {d['id']} | {short} | {d.get('expected_doc','—')} | {hit} | {top1} | {docs} |")
    return "\n".join(lines)


def _quality_table(quality_result: dict) -> str:
    """生成回答质量评分表（Markdown），适配新 schema。"""
    if quality_result.get("error"):
        return f"> ⚠️ {quality_result['error']}"

    lines = [
        "| 题号 | 准确性 | 有引用? | 有幻觉? | judge 评语（缩略）|",
        "|------|-------|--------|--------|----------------|",
    ]
    for d in quality_result.get("details", []):
        if "scores" not in d:
            err_msg = d.get("rag_error") or d.get("judge_error") or "解析失败"
            lines.append(f"| {d['id']} | — | — | — | ❌ {err_msg[:30]} |")
            continue
        s      = d["scores"]
        acc    = s.get("accuracy_score", "—")
        cit    = "✅" if s.get("has_citation") else "❌"
        hal    = "⚠️是" if s.get("hallucination_detected") else "✅否"
        reason = s.get("reason", "")
        reason = reason[:30] + "…" if len(reason) > 30 else reason
        lines.append(f"| {d['id']} | {acc}/5 | {cit} | {hal} | {reason} |")
    return "\n".join(lines)


def _problems_section(routing: dict, retrieval: dict, quality: dict) -> str:
    """列出所有评估中表现不好的具体 case。"""
    lines = []

    # 路由误判
    routing_errors = []
    for m, stats in routing["modes"].items():
        for e in stats.get("errors", []):
            routing_errors.append((m, e))
    if routing_errors:
        lines.append("### 🔴 路由误判")
        lines.append("")
        for mode, e in routing_errors:
            lines.append(f"- **[{e['id']}]** `{mode}` 模式：预期 `{e['expected']}`，判断为 `{e['got']}`")
            lines.append(f"  - 问题：{e['question']}")
            lines.append(f"  - 模型依据：{e['reason']}")
            lines.append(f"  - 是否降级：{'是' if e['fallback'] else '否'}")
            lines.append(f"  - **建议**：检查 router.py 的 prompt 对该意图的描述是否足够清晰")
            lines.append("")

    # 检索未命中
    if retrieval.get("misses"):
        lines.append("### 🟡 检索未命中")
        lines.append("")
        for m in retrieval["misses"]:
            lines.append(f"- **[{m['id']}]** 期望文档《{m.get('expected_doc','?')}》未在 top-k 中出现")
            lines.append(f"  - 问题：{m['question']}")
            docs = "、".join(
                f"《{r['doc_name']}》({r['score']:.3f})"
                for r in m.get("retrieved_docs", [])[:3]
            ) if m.get("retrieved_docs") else "（无数据）"
            lines.append(f"  - 实际检索到：{docs}")
            lines.append(f"  - **诊断方向**：可能是 embedding 语义区分度不足，或 chunk 边界把关键词切断；"
                         "建议缩小 CHUNK_SIZE 或换更强的 embedding 模型")
            lines.append("")

    # 低分回答（准确性 ≤ 2 或检出幻觉）
    low_quality = []
    for d in quality.get("details", []):
        s = d.get("scores", {})
        if not s:
            continue
        if s.get("accuracy_score", 5) <= 2 or s.get("hallucination_detected"):
            low_quality.append(d)
    if low_quality:
        lines.append("### 🔴 回答质量问题（准确性≤2 或检出幻觉）")
        lines.append("")
        for d in low_quality:
            s = d["scores"]
            acc_str = f"准确性 {s['accuracy_score']}/5"
            hal_str = "⚠️检出幻觉" if s.get("hallucination_detected") else ""
            flags   = "  ".join(filter(None, [acc_str, hal_str]))
            lines.append(f"- **[{d['id']}]** {flags}")
            lines.append(f"  - 问题：{d['question']}")
            lines.append(f"  - judge 评语：{s.get('reason','')}")
            lines.append(f"  - **建议**：检查 generate.py 的引用 prompt；若是检索未命中导致，优先修 embedding")
            lines.append("")

    if not lines:
        lines.append("> ✅ 本次评估未发现明显问题。")

    return "\n".join(lines)


def generate_ollama_conclusion(
    routing: dict,
    retrieval: dict,
    quality: dict,
    model: str = None,
) -> str:
    """
    调用本地 Ollama 模型，基于评估数据自动生成结论性总结。
    不需要任何 API Key，完全本地运行。
    """
    import httpx
    from evaluate import _OLLAMA_BASE_URL, JUDGE_MODEL

    if model is None:
        model = JUDGE_MODEL  # 复用裁判模型即可

    summary_data = {
        "routing": {
            m: {"accuracy": v["accuracy"], "error_ids": [e["id"] for e in v["errors"]]}
            for m, v in routing["modes"].items()
        },
        "retrieval": {
            "recall_at_k": retrieval["recall_at_k"],
            "top_k": retrieval["top_k"],
            "miss_ids": [m["id"] for m in retrieval.get("misses", [])],
        },
        "quality": {
            "avg_scores": quality.get("avg_scores", {}),
            "scored":     quality.get("scored", 0),
        },
    }

    prompt = (
        "你是一位 RAG 系统评估专家，请基于以下评估数据，用中文写一段 150 字以内的结论性总结。\n"
        "要求：\n"
        "1. 点出各模块的关键指标数字\n"
        "2. 指出最需要优化的 1-2 个问题点并给出建议\n"
        "3. 专业简洁，给出分析判断而非数字堆砌\n\n"
        f"评估数据：\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}"
    )

    client = httpx.Client(trust_env=False)
    resp = client.post(
        f"{_OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.3},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def build_report(
    routing: dict,
    retrieval: dict,
    quality: dict,
    run_time: str,
    skip_quality: bool = False,
    claude_conclusion: str = "",
) -> str:
    """把三个模块的结果拼成完整的 Markdown 报告。"""
    modes = list(routing["modes"].keys())

    # 路由准确率摘要
    routing_summary = "  ".join(
        f"**{m}**: {v['correct']}/{v['total']} ({v['accuracy']:.0%})"
        for m, v in routing["modes"].items()
    )

    # 质量平均分摘要（适配新 schema）
    avg = quality.get("avg_scores", {})
    if avg:
        cit_pct  = f"{avg.get('citation_rate', 0):.0%}"
        hal_pct  = f"{avg.get('hallucination_rate', 0):.0%}"
        quality_summary = (
            f"准确性 **{avg.get('accuracy','—')}/5**"
            f"  引用率 {cit_pct}"
            f"  幻觉率 {hal_pct}（越低越好）"
        )
    else:
        quality_summary = "（未运行）"

    report = f"""# 银行合规知识助手 — 自动评估报告

> 生成时间：{run_time}
> 数据集：eval_dataset.json（{retrieval['total'] + len([d for d in routing['per_item'] if d['expected'] != '知识型' and d['expected'] is not None])} 条有效题 + 模糊题）

---

## 一、总体概览

| 评估模块 | 指标 | 结果 |
|---------|------|------|
| 意图路由 | 准确率 | {routing_summary} |
| RAG 检索 | Recall@{retrieval['top_k']} | **{retrieval['recall_at_k']:.1%}**（{retrieval['hits']}/{retrieval['total']}） |
| 回答质量 | 平均分（1-5） | {quality_summary} |

---

## 二、意图路由准确率

{_routing_table(routing)}

> ↩ = 触发降级兜底（tool_calling 模式未调用工具，自动回退 json_parse）
> — = 模糊题，不计入分母

---

## 三、RAG 检索准确率（Recall@{retrieval['top_k']}）

{_retrieval_table(retrieval)}

---

## 四、回答质量评分（LLM-as-judge）

{"跳过（运行时使用 --no-quality 标志）" if skip_quality else _quality_table(quality)}

> 评分维度：准确性 1-5 分 / 有无引用 / 有无幻觉
> 裁判模型：{quality.get('judge_model', JUDGE_MODEL)}（与生成模型不同，避免 self-preference bias）
> ⚠️ 建议人工抽查 3-5 条分数，核实裁判判断是否合理，尤其关注极端分（1分/5分）

---

## 五、问题清单（需重点排查的 Case）

{_problems_section(routing, retrieval, quality)}

---

## 六、自动生成结论

{claude_conclusion if claude_conclusion else '> （使用 --no-quality 标志时跳过，不跳过则由本地 Ollama 自动生成）'}

---

## 七、各模块关系与优化指引

```
用户问题
   │
   ▼ [模块A 路由] 准确率直接决定后续流程是否走对分支
   │   错误 → 后续的检索和回答都会跑偏，应优先修路由
   │
   ├── 知识型 → [模块B 检索] Recall@k 衡量能否找到正确文档
   │             未命中 → 回答必然不准，应调 chunk_size 或 embedding 模型
   │
   └── (正确检索后) → [模块C 回答] LLM-as-judge 衡量生成质量
                        低分 → 调整 generate.py 的 prompt 模板
```

| 发现的问题 | 对应优化方向 |
|-----------|-------------|
| 路由误判率高 | 修改 `router.py` 的 prompt，明确各意图的描述边界 |
| 检索 Recall 低 | 缩小 `CHUNK_SIZE`（减少噪音），或换 `EMBEDDING_MODE=openai` |
| 回答引用分低 | 强化 `generate.py` prompt 中对引用格式的要求 |
| 回答幻觉分低 | 在 prompt 中加"只能基于材料回答，不得推断" |
"""
    return report


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="运行评估并生成 report.md")
    parser.add_argument("--no-quality", action="store_true", help="跳过 LLM-as-judge 回答质量评估（节省时间）")
    parser.add_argument("--top-k", type=int, default=3, help="检索 top-k（默认3）")
    args = parser.parse_args()

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═' * 60}")
    print(f"  🧪 银行合规助手 — 自动评估系统")
    print(f"  运行时间：{run_time}")
    print(f"{'═' * 60}")

    # 加载数据集
    dataset = load_dataset()
    print(f"\n📂 数据集：{len(dataset)} 条（知识型 {sum(1 for d in dataset if d['expected_intent']=='知识型')}，"
          f"查询型 {sum(1 for d in dataset if d['expected_intent']=='查询型')}，"
          f"操作型 {sum(1 for d in dataset if d['expected_intent']=='操作型')}，"
          f"模糊 {sum(1 for d in dataset if d['expected_intent'] is None)}）")

    # 初始化 Embedding 模型（三个模块共用）
    print("\n🔮 初始化 Embedding 模型...")
    embed_fn = get_embedding_function()

    # ── 模块A：意图路由准确率 ──────────────────────────────────
    t0 = time.time()
    routing_result = eval_routing(dataset)
    print(f"\n  ✅ 路由评估完成（{time.time()-t0:.1f}s）")

    # ── 模块B：RAG 检索准确率 ──────────────────────────────────
    t0 = time.time()
    retrieval_result = eval_retrieval(dataset, embed_fn, DB_PATH, top_k=args.top_k)
    print(f"\n  ✅ 检索评估完成（{time.time()-t0:.1f}s）")

    # ── 模块C：回答质量评估（本地 Ollama，无需 API Key）────────
    if args.no_quality:
        print(f"\n  ⏭️  跳过回答质量评估（--no-quality）")
        quality_result = {"error": "--no-quality", "details": [], "avg_scores": {}, "judge_model": "—"}
    else:
        t0 = time.time()
        quality_result = eval_answer_quality(dataset, embed_fn, DB_PATH)
        print(f"\n  ✅ 质量评估完成（{time.time()-t0:.1f}s）")

    # ── 保存原始结果 ───────────────────────────────────────────
    raw = {"routing": routing_result, "retrieval": retrieval_result, "quality": quality_result}
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"\n💾 原始结果已保存：{RAW_PATH.name}")

    # ── 生成 Ollama 结论（无需 API Key）──────────────────────────
    ollama_conclusion = ""
    if not args.no_quality:
        print(f"\n🤖 调用 Ollama 生成评估结论（{JUDGE_MODEL}）...")
        try:
            ollama_conclusion = generate_ollama_conclusion(
                routing_result, retrieval_result, quality_result
            )
            print(f"  {ollama_conclusion[:100]}…")
        except Exception as e:
            print(f"  ⚠️  结论生成失败：{e}")

    # ── 生成 report.md ─────────────────────────────────────────
    report_md = build_report(
        routing_result, retrieval_result, quality_result,
        run_time=run_time,
        skip_quality=args.no_quality,
        claude_conclusion=ollama_conclusion,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n{'═' * 60}")
    print(f"  📄 报告已生成：{REPORT_PATH.name}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
