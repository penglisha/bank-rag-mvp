"""
run_tests.py — 自动化路由测试
对 test_questions.txt 中的 11 道题，分别用 json_parse 和 tool_calling 路由，
输出意图判断对比表。
"""
import sys, os, time
sys.path.insert(0, "src")

# 11 道测试题及预期意图（模糊题预期为 None）
TESTS = [
    ("Q01", "知识型", "信用卡逾期30天会有什么后果？"),
    ("Q02", "知识型", "理财产品的风险等级是怎么划分的？R3级适合什么类型的投资者？"),
    ("Q03", "知识型", "个人贷款提前还款需要支付违约金吗？什么时候不需要？"),
    ("Q04", "知识型", "客户投诉银行之后，银行需要在多少个工作日内回复？"),
    ("Q05", "查询型", "帮我查一下 C002 客户的账户情况"),
    ("Q06", "查询型", "C004 这个客户逾期了几天？信用评分是多少？"),
    ("Q07", "查询型", "查询客户 C001 目前还有多少可用贷款额度"),
    ("Q08", "操作型", "C002 客户逾期了45天，帮我提交一个逾期罚息减免申请"),
    ("Q09", "操作型", "客户 C004 信用卡额度快用完了，她申请提高信用卡额度，请帮我提交申请"),
    ("Q10", None,    "C002 客户最近有没有逾期？如果逾期了按规定要怎么处理？"),
    ("Q11", None,    "账户 C003 上个月还款正常，但他想申请一个利率优惠，帮我走一下流程"),
]

MODES = ["json_parse", "tool_calling"]
INTENT_EMOJI = {"知识型": "📚", "查询型": "🔍", "操作型": "✍️ ", None: "❓"}

def run_one(question, mode):
    """对单题跑一次路由，返回 (intent, reason, fallback, elapsed_ms)。"""
    os.environ["ROUTING_MODE"] = mode
    # 重新导入确保环境变量生效（router 在模块层读取了 ROUTING_MODE）
    import importlib, router as rt
    importlib.reload(rt)

    t0 = time.time()
    result = rt.route(question)
    elapsed = int((time.time() - t0) * 1000)
    return result["intent"], result["reason"], result["fallback"], elapsed

def correct_mark(got, expected):
    if expected is None:
        return "—"          # 模糊题不判对错
    return "✅" if got == expected else "❌"

# ── 主流程 ────────────────────────────────────────────────────
print("\n" + "═" * 68)
print("  🧪 路由方式对比测试（qwen2.5:7b）")
print("  模式：json_parse  vs  tool_calling")
print("═" * 68)

results = {}   # {qid: {mode: (intent, correct, fallback, ms)}}

for qid, expected, question in TESTS:
    results[qid] = {}
    short_q = question[:38] + "…" if len(question) > 38 else question
    print(f"\n{qid} [{expected or '模糊'}] {short_q}")

    for mode in MODES:
        intent, reason, fallback, ms = run_one(question, mode)
        mark = correct_mark(intent, expected)
        fb = " ↩降级" if fallback else ""
        print(f"  {mode:<14}: {INTENT_EMOJI.get(intent, '?')} {intent:<5}  {mark}{fb}  ({ms}ms)")
        print(f"               依据: {reason[:55]}")
        results[qid][mode] = (intent, mark, fallback, ms)

# ── 汇总表 ────────────────────────────────────────────────────
print("\n\n" + "═" * 68)
print("  📊 汇总对比表")
print("═" * 68)
header = f"{'题号':<5} {'预期':<5}  {'json_parse结果':<8}{'✓':<3}  {'tool_calling结果':<9}{'✓':<3}"
print(header)
print("─" * 60)

correct = {"json_parse": 0, "tool_calling": 0}
total_scored = 0

for qid, expected, question in TESTS:
    jp  = results[qid]["json_parse"]
    tc  = results[qid]["tool_calling"]
    jp_fb  = "↩" if jp[2] else " "
    tc_fb  = "↩" if tc[2] else " "
    exp_str = expected if expected else "模糊"
    print(f"{qid:<5} {exp_str:<5}  {jp[0]:<8}{jp_fb} {jp[1]:<3}  {tc[0]:<10}{tc_fb} {tc[1]:<3}")
    if expected:
        total_scored += 1
        if jp[1] == "✅": correct["json_parse"] += 1
        if tc[1] == "✅": correct["tool_calling"] += 1

print("─" * 60)
print(f"{'正确率':<11} {correct['json_parse']}/{total_scored} ({correct['json_parse']*100//total_scored}%)      "
      f"{correct['tool_calling']}/{total_scored} ({correct['tool_calling']*100//total_scored}%)")
print()
print("↩ = 触发降级兜底   — = 模糊题不计分")
print("═" * 68)
