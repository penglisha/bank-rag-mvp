"""
tools.py — Agent 工具模块

定义 Agent 可以调用的"工具"。在真实银行系统中，这些函数会连接实际数据库和业务系统。
这里用 Python 字典模拟客户数据，接口契约（函数签名和返回结构）与真实系统保持一致，
方便后续一键替换为真实实现。

【包含两类工具】
1. query_customer_info    —— 只读查询，可直接返回结果
2. submit_operation_request —— 写操作，只生成审批申请，不直接执行

【为什么操作型要"生成申请而非直接执行"？】
金融场景下，涉及资金流转或账户状态变更的操作必须经过人工复核。
这是"Human-in-the-loop（人在环路中）"设计原则：
AI 负责识别意图、整理信息、生成标准申请单，
人工负责最终审批决策。
这样既发挥了 AI 的效率，又保留了人工兜底的安全边际。
"""

from datetime import datetime
from typing import Optional
import random
import string

# ─── 模拟客户数据库 ──────────────────────────────────────────
# 包含 5 个客户，覆盖：正常/轻度逾期/中度逾期/满额使用/零使用 等不同状态
CUSTOMER_DB: dict[str, dict] = {
    "C001": {
        "name": "张伟",
        "product_type": "消费贷款",
        "loan_limit": 500_000,     # 贷款额度（元）
        "used_amount": 120_000,    # 已用额度（元）
        "overdue": False,
        "overdue_days": 0,
        "credit_score": 720,
    },
    "C002": {
        "name": "李娜",
        "product_type": "信用卡",
        "loan_limit": 200_000,
        "used_amount": 200_000,    # 已用满额
        "overdue": True,
        "overdue_days": 45,        # M2 中度逾期（31-60天）
        "credit_score": 580,
    },
    "C003": {
        "name": "王建国",
        "product_type": "经营贷款",
        "loan_limit": 1_000_000,
        "used_amount": 350_000,
        "overdue": False,
        "overdue_days": 0,
        "credit_score": 780,
    },
    "C004": {
        "name": "陈小燕",
        "product_type": "信用卡",
        "loan_limit": 50_000,
        "used_amount": 48_000,
        "overdue": True,
        "overdue_days": 12,        # M1 轻度逾期（1-30天）
        "credit_score": 630,
    },
    "C005": {
        "name": "刘强",
        "product_type": "消费贷款",
        "loan_limit": 300_000,
        "used_amount": 0,          # 尚未使用
        "overdue": False,
        "overdue_days": 0,
        "credit_score": 750,
    },
}


def _overdue_level(days: int) -> str:
    """根据逾期天数返回逾期等级描述（对应信用卡逾期处理规定的 M1-M3 分类）。"""
    if days == 0:
        return "正常"
    if days <= 30:
        return f"逾期 {days} 天（M1 轻度逾期）"
    if days <= 60:
        return f"逾期 {days} 天（M2 中度逾期）"
    if days <= 90:
        return f"逾期 {days} 天（M3 重度逾期）"
    return f"逾期 {days} 天（M4+ 严重逾期）"


def query_customer_info(customer_id: str) -> dict:
    """
    查询客户账户信息（只读操作）。

    参数：
        customer_id: 客户编号，如 "C001"（大小写不敏感）

    返回：
        success=True 时包含完整客户信息；
        success=False 时包含 error 说明和可用的客户编号列表。
    """
    cid = customer_id.strip().upper()

    if cid not in CUSTOMER_DB:
        return {
            "success": False,
            "error": f"未找到客户编号 [{cid}] 的记录",
            "hint": f"当前数据库中的客户编号：{', '.join(CUSTOMER_DB.keys())}",
        }

    raw = CUSTOMER_DB[cid]
    available = raw["loan_limit"] - raw["used_amount"]

    return {
        "success": True,
        "customer_id": cid,
        "name": raw["name"],
        "product_type": raw["product_type"],
        "loan_limit": raw["loan_limit"],
        "used_amount": raw["used_amount"],
        "available_amount": available,
        "loan_limit_str": f"{raw['loan_limit']:,} 元",
        "used_amount_str": f"{raw['used_amount']:,} 元",
        "available_amount_str": f"{available:,} 元",
        "overdue_status": _overdue_level(raw["overdue_days"]),
        "credit_score": raw["credit_score"],
    }


def submit_operation_request(
    operation_type: str,
    customer_id: str,
    detail: str,
) -> dict:
    """
    提交操作申请（不直接执行，仅生成待人工审批的申请单）。

    【设计原则】
    AI 不直接修改任何数据——它只是把用户意图整理成标准格式的申请单，
    交由人工在后台审批系统中处理。这是金融合规场景的硬性要求。

    参数：
        operation_type: 操作类型，如"逾期减免申请""提额申请""账户解冻"
        customer_id   : 客户编号
        detail        : 操作详情描述
    """
    cid = customer_id.strip().upper()
    customer = CUSTOMER_DB.get(cid)
    customer_name = customer["name"] if customer else "（客户编号未知）"

    # 生成唯一申请单号（模拟）
    req_id = "REQ-" + "".join(random.choices(string.digits, k=8))
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "success": True,
        "request_id": req_id,
        "status": "待人工审批",
        "operation_type": operation_type,
        "customer_id": cid,
        "customer_name": customer_name,
        "detail": detail,
        "created_at": created_at,
        "message": f"已生成申请单 [{req_id}]，等待人工审批",
    }

    # 打印申请详情（模拟写入后台审批系统）
    print(f"\n  ┌─ 操作申请已提交 ──────────────────────────────")
    print(f"  │  申请单号：{req_id}")
    print(f"  │  操作类型：{operation_type}")
    print(f"  │  客户信息：{cid} / {customer_name}")
    print(f"  │  操作详情：{detail}")
    print(f"  │  提交时间：{created_at}")
    print(f"  │  申请状态：⏳ 待人工审批")
    print(f"  └──────────────────────────────────────────────")

    return result


def format_customer_info(info: dict) -> str:
    """将 query_customer_info 的返回值格式化为可读字符串，供打印和传给 LLM 使用。"""
    if not info.get("success"):
        return f"❌ 查询失败：{info.get('error', '未知错误')}\n   {info.get('hint', '')}"

    overdue_flag = "⚠️ " if info["overdue_status"] != "正常" else "✅ "

    return (
        f"客户编号：{info['customer_id']}\n"
        f"姓    名：{info['name']}\n"
        f"产品类型：{info['product_type']}\n"
        f"贷款额度：{info['loan_limit_str']}\n"
        f"已用额度：{info['used_amount_str']}\n"
        f"可用额度：{info['available_amount_str']}\n"
        f"逾期状态：{overdue_flag}{info['overdue_status']}\n"
        f"信用评分：{info['credit_score']}"
    )
