"""
agent_main.py — Agent 交互入口

在原有纯 RAG 流程（main.py）基础上新增 Agent 能力：
  意图路由 → 分支执行（RAG / 客户查询 / 操作申请）

与 main.py 的关系：
  - main.py：保持不变，仍可独立运行，只做纯 RAG
  - agent_main.py：复用 main.py 的 RAG 初始化和查询函数，
    在此之上加入意图识别和工具调用层

运行方式：
  python src/agent_main.py                          # 默认 json_parse 路由
  ROUTING_MODE=tool_calling python src/agent_main.py  # 切换到 tool_calling 路由
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 将 src 目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent))

# 复用已有 RAG 模块
from main import initialize_system, run_query, CHUNK_SIZE, TOP_K

# Agent 新增模块
from router import route, ROUTING_MODE, ROUTER_MODEL, INTENT_KNOWLEDGE, INTENT_QUERY, INTENT_OPERATION
from tools import query_customer_info, submit_operation_request, format_customer_info


# ══════════════════════════════════════════════════════════════
# 各意图分支的执行函数
# ══════════════════════════════════════════════════════════════

def handle_knowledge(question: str, embed_fn) -> str:
    """
    知识型 → 走 RAG 流程
    直接复用原有 run_query，检索向量库并调用 LLM 生成带引用的回答。
    """
    print("\n  📚 【知识型】调用 RAG 知识库检索...")
    return run_query(question, embed_fn)


def handle_query(question: str, params: dict) -> str:
    """
    查询型 → 调用客户信息工具
    从 params 中取 customer_id；若模型未提取到，尝试从问题文本中识别，
    最后实在找不到则提示用户补充。
    """
    customer_id = params.get("customer_id", "").strip()

    # 兜底：如果模型没有提取出 customer_id，从文本中找 C+数字 格式
    if not customer_id:
        import re
        match = re.search(r"[Cc]\d{3}", question)
        if match:
            customer_id = match.group(0).upper()

    if not customer_id:
        return (
            "❓ 请提供客户编号（格式如 C001）才能查询账户信息。\n"
            "   当前可查询的客户：C001（张伟）/ C002（李娜）/ "
            "C003（王建国）/ C004（陈小燕）/ C005（刘强）"
        )

    print(f"\n  🔍 【查询型】调用客户信息工具，查询客户 {customer_id}...")
    info = query_customer_info(customer_id)
    return format_customer_info(info)


def handle_operation(question: str, params: dict) -> str:
    """
    操作型 → 提交操作申请（不直接执行，生成待审批申请单）
    从 params 中取参数；缺失字段有降级处理，不会崩溃。
    """
    operation_type = params.get("operation_type", "").strip()
    customer_id    = params.get("customer_id", "").strip().upper()
    detail         = params.get("detail", "").strip()

    # 操作类型兜底：从问题中提取
    if not operation_type:
        # 常见操作关键词映射
        kw_map = {
            "提额": "信用卡提额申请",
            "额度": "信用卡提额申请",
            "减免": "逾期减免申请",
            "罚息": "逾期减免申请",
            "解冻": "账户解冻申请",
            "利率": "利率优惠申请",
        }
        for kw, op in kw_map.items():
            if kw in question:
                operation_type = op
                break
        if not operation_type:
            operation_type = "业务操作申请"

    # customer_id 兜底：从文本中找
    if not customer_id:
        import re
        match = re.search(r"[Cc]\d{3}", question)
        if match:
            customer_id = match.group(0).upper()

    if not customer_id:
        return (
            "❓ 操作申请需要提供客户编号（格式如 C001）。\n"
            "   请重新描述，例如：'为 C002 客户提交逾期减免申请'"
        )

    # detail 兜底：用原始问题作为详情
    if not detail:
        detail = question

    print(f"\n  ✍️  【操作型】提交操作申请...")
    result = submit_operation_request(operation_type, customer_id, detail)
    return result["message"]


# ══════════════════════════════════════════════════════════════
# 主交互函数
# ══════════════════════════════════════════════════════════════

def agent_query(question: str, embed_fn) -> None:
    """
    执行一次完整的 Agent 查询，打印每一步中间过程：
    ① 路由方式 + 意图判断
    ② 对应分支执行
    ③ 最终结果
    """
    print(f"\n{'═' * 62}")
    print(f"  ❓ 用户问题：{question}")
    print(f"{'═' * 62}")

    # ── 步骤1：意图路由 ──────────────────────────────────────
    print(f"\n🧭 【意图路由】模式：{ROUTING_MODE}  模型：{ROUTER_MODEL}")
    route_result = route(question)

    intent   = route_result["intent"]
    reason   = route_result["reason"]
    params   = route_result["params"]
    fallback = route_result["fallback"]

    fallback_note = " [兜底降级]" if fallback else ""
    print(f"   意图类型：{intent}{fallback_note}")
    print(f"   判断依据：{reason}")
    if params:
        print(f"   提取参数：{params}")

    # ── 步骤2：分支执行 ──────────────────────────────────────
    if intent == INTENT_KNOWLEDGE:
        answer = handle_knowledge(question, embed_fn)
    elif intent == INTENT_QUERY:
        answer = handle_query(question, params)
    elif intent == INTENT_OPERATION:
        answer = handle_operation(question, params)
    else:
        answer = f"未知意图类型 [{intent}]，请重新描述您的问题。"

    # ── 步骤3：打印结果 ──────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("💡 执行结果")
    print(f"{'─' * 62}")
    print(answer)
    print(f"{'─' * 62}")


def print_welcome(routing_mode: str):
    """打印欢迎界面和示例问题。"""
    print("\n" + "═" * 62)
    print("  🏦 银行合规知识助手 —— Agent 模式")
    print(f"  路由方式：{routing_mode}  |  路由模型：{ROUTER_MODEL}")
    print("═" * 62)
    print("\n  支持三类问题：")
    print("  📚 知识型 → 合规知识库检索（RAG）")
    print("  🔍 查询型 → 客户账户信息查询")
    print("  ✍️  操作型 → 提交人工审批申请")
    print("\n  示例问题：")
    examples = [
        ("知识型", "信用卡逾期30天会有什么后果？"),
        ("知识型", "理财产品 R3 级别适合什么类型的投资者？"),
        ("查询型", "帮我查一下 C002 客户的账户情况"),
        ("查询型", "C004 这个客户逾期了几天？"),
        ("操作型", "C002 逾期了，帮我提交一个逾期减免申请"),
        ("操作型", "客户 C004 申请提高信用卡额度，帮我提交申请"),
        ("模糊型", "C002 最近逾期情况怎么样，逾期了要怎么处理？"),
    ]
    for tag, q in examples:
        print(f"    [{tag}] {q}")
    print()
    print("  输入 quit 退出 | rebuild 重建向量库 | mode 切换路由方式")
    print()


def main():
    """Agent 模式主入口。"""
    # 初始化 RAG 系统（切块 + 向量库）
    embed_fn = initialize_system()

    # 打印欢迎界面
    print_welcome(ROUTING_MODE)

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

        if user_input.lower() == "mode":
            current = os.getenv("ROUTING_MODE", "json_parse")
            other   = "tool_calling" if current == "json_parse" else "json_parse"
            print(f"\n  当前路由模式：{current}")
            print(f"  切换方式：export ROUTING_MODE={other}，然后重启程序")
            print(f"  或在 router.py 中直接修改 ROUTING_MODE 常量")
            continue

        agent_query(user_input, embed_fn)


if __name__ == "__main__":
    main()
