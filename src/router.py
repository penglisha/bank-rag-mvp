"""
router.py — 意图路由模块

根据用户输入，判断应该走哪条处理分支：
  - 知识型 → RAG 知识库检索
  - 查询型 → 客户信息工具
  - 操作型 → 操作申请工具

支持两种路由方式，通过 ROUTING_MODE 配置切换：
  - json_parse   （默认）：提示词要求模型输出 JSON，Python 解析，带降级兜底
  - tool_calling          ：原生工具调用，让模型自己决定调哪个工具
"""

import os
import json
import re
from typing import Optional
import httpx
from openai import OpenAI

# ─── 配置区 ──────────────────────────────────────────────────
# 修改 ROUTING_MODE 或在 .env 中设置同名环境变量来切换路由方式
ROUTING_MODE: str = os.getenv("ROUTING_MODE", "json_parse")
# json_parse：稳定，适合本地小模型；tool_calling：依赖模型原生支持

ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "qwen2.5:7b")
# 本机已有：qwen2.5:7b / gemma4-hermes:latest（tool calling 支持更好）
# 如 qwen2.5 tool calling 不稳定，可改为 gemma4-hermes:latest

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# ─────────────────────────────────────────────────────────────

# 意图类型常量
INTENT_KNOWLEDGE = "知识型"
INTENT_QUERY     = "查询型"
INTENT_OPERATION = "操作型"

# 兜底意图：当路由失败时，默认走知识型 RAG（最安全）
FALLBACK_INTENT = INTENT_KNOWLEDGE


# ─── 返回类型说明 ─────────────────────────────────────────────
# route() 统一返回以下结构的字典：
# {
#   "intent"  : "知识型" | "查询型" | "操作型",
#   "reason"  : str,   # 模型给出的判断依据（一句话）
#   "params"  : dict,  # 意图相关参数（见各意图说明）
#   "mode"    : str,   # 实际使用的路由方式
#   "fallback": bool,  # True 表示发生了异常并降级到兜底逻辑
# }
#
# params 字段约定：
#   知识型  → {}（不需要额外参数，agent_main.py 直接用原始问题做 RAG）
#   查询型  → {"customer_id": "C001"}
#   操作型  → {"operation_type": "xxx", "customer_id": "C001", "detail": "xxx"}
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 工具定义（供方式A tool_calling 使用）
# ══════════════════════════════════════════════════════════════

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "搜索银行合规知识库，回答关于银行产品规定、法规条款、"
                "业务流程、利率费用、风险等级等合规知识问题。"
                "适用于不涉及特定客户的通用性问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的原始问题，用于在知识库中检索相关内容",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_customer",
            "description": (
                "查询特定客户的账户信息，包括贷款额度、已用额度、"
                "可用额度、逾期状态、信用评分等。"
                "仅当用户明确提到客户编号（如C001）或客户姓名时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "客户编号，格式如 C001、C002 等（大小写不敏感）",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_operation",
            "description": (
                "提交需要人工审批的操作申请，如逾期减免、信用卡提额、"
                "账户解冻、利率优惠等。"
                "AI 不直接执行操作，只生成申请单等待人工审批。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_type": {
                        "type": "string",
                        "description": "操作类型，如：逾期减免申请、信用卡提额、账户解冻、利率优惠申请",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "客户编号，如 C001",
                    },
                    "detail": {
                        "type": "string",
                        "description": "操作详情描述，包含业务背景和具体诉求",
                    },
                },
                "required": ["operation_type", "customer_id", "detail"],
            },
        },
    },
]

# 工具名称 → 意图类型的映射
_TOOL_TO_INTENT = {
    "rag_search":       INTENT_KNOWLEDGE,
    "query_customer":   INTENT_QUERY,
    "submit_operation": INTENT_OPERATION,
}


# ══════════════════════════════════════════════════════════════
# 方式A：原生 Tool Calling
# ══════════════════════════════════════════════════════════════

def _route_tool_calling(question: str) -> dict:
    """
    【方式A：原生工具调用（Tool Calling）】

    原理：
    向模型发送"工具定义"列表，模型在训练时学习了工具调用格式，
    能直接输出结构化的 tool_calls（包含工具名和参数）。
    这是一种"模型自主决策"的方式——我们定义工具集合，模型自己选。

    优点：
    - 参数提取准确（模型直接生成结构化 JSON，不需要额外解析）
    - 代码逻辑简洁，无需处理 JSON 解析异常
    - 工具描述即文档，维护方便

    缺点：
    - 依赖模型是否经过工具调用微调（部分本地小模型支持不稳定）
    - 若模型选择不调用任何工具（直接回答），需要兜底处理
    - 调试时不如 JSON 直观（返回结构更复杂）

    适用场景：
    - 使用支持 function calling 的模型（hermes3、qwen2.5、llama3.1 等）
    - 需要精确提取多个参数的场景（如 submit_operation 的三个字段）
    - 云端 API（OpenAI、Claude）通常优先选此方式
    """
    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        http_client=httpx.Client(trust_env=False),  # 绕过系统代理
    )

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是银行合规助手的意图识别模块。"
                        "根据用户输入，选择最合适的工具处理。"
                        "必须调用一个工具，不要直接回答问题。"
                    ),
                },
                {"role": "user", "content": question},
            ],
            tools=_TOOLS,
            tool_choice="required",   # 强制必须调用工具，避免模型直接回答
        )
    except Exception as e:
        return _fallback_result(f"tool_calling API 调用失败：{e}")

    msg = response.choices[0].message

    # 没有 tool_calls（模型未调用工具）→ 降级到 json_parse
    if not msg.tool_calls:
        print(f"  ⚠️  [tool_calling] 模型未调用工具，降级到 json_parse 模式")
        result = _route_json_parse(question)
        result["fallback"] = True
        return result

    # 取第一个工具调用（通常只有一个）
    call = msg.tool_calls[0]
    tool_name = call.function.name
    intent = _TOOL_TO_INTENT.get(tool_name, FALLBACK_INTENT)

    try:
        params = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        params = {}

    return {
        "intent":   intent,
        "reason":   f"模型选择调用工具 [{tool_name}]",
        "params":   params,
        "mode":     "tool_calling",
        "fallback": False,
    }


# ══════════════════════════════════════════════════════════════
# 方式B：JSON 输出 + 手动解析
# ══════════════════════════════════════════════════════════════

_JSON_PARSE_SYSTEM = """\
你是银行合规助手的意图识别模块。根据用户输入判断意图类型，严格按格式输出JSON。

意图类型和对应的params格式：
- 知识型：询问银行规定/法规/流程/产品说明等合规知识 → params为空对象{}
- 查询型：查询特定客户的账户信息、逾期状态等 → params包含{"customer_id": "C00X"}
- 操作型：申请办理需要人工审批的操作（提额/减免/解冻等）→ params包含{"operation_type":"xxx","customer_id":"C00X","detail":"xxx"}

输出格式（只输出JSON，不要有任何其他内容）：
{"intent": "知识型/查询型/操作型", "reason": "判断依据一句话", "params": {...}}"""


def _route_json_parse(question: str) -> dict:
    """
    【方式B：JSON 输出 + 手动解析（降级稳妥方案）】

    原理：
    不依赖模型的工具调用能力，而是在 prompt 中直接告诉模型
    "只输出 JSON，格式如下"，然后用 Python 解析结果。
    使用 Ollama 的 format:"json" 参数（结构化输出模式），
    强制模型输出合法 JSON，大幅提升解析成功率。

    优点：
    - 不依赖 function calling 微调，兼容性更强（几乎所有模型都能跟指令输出 JSON）
    - 调试直观：直接看 prompt 和 raw 输出即可理解模型在想什么
    - 可以加丰富的异常处理和降级逻辑
    - format:"json" 模式确保输出合法 JSON，避免无引号键名等问题

    缺点：
    - params 提取依赖 prompt 设计，偶尔提取不完整（尤其复杂操作型）
    - 多加了一层 JSON 解析代码
    - format:"json" 是 Ollama 原生 API 特性，需用 /api/chat 而非 OpenAI 兼容端点

    适用场景：
    - 本地小模型（7B 以下），tool calling 格式支持不稳定的情况
    - 快速原型和调试阶段
    - 需要对模型推理过程有完全控制权的场景
    """
    http_client = httpx.Client(trust_env=False)  # 绕过系统代理

    try:
        response = http_client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": ROUTER_MODEL,
                "format": "json",    # Ollama 结构化输出模式：强制合法 JSON
                "stream": False,
                "messages": [
                    {"role": "system", "content": _JSON_PARSE_SYSTEM},
                    {"role": "user",   "content": question},
                ],
                "options": {"temperature": 0},  # 温度为0，让判断更确定性
            },
            timeout=30.0,
        )
        response.raise_for_status()
        raw_text = response.json()["message"]["content"]
    except Exception as e:
        return _fallback_result(f"json_parse API 调用失败：{e}")

    # 解析 JSON
    parsed = _try_parse_json(raw_text)
    if parsed is None:
        print(f"  ⚠️  [json_parse] JSON 解析失败，原始输出：{repr(raw_text[:100])}")
        return _fallback_result(f"模型输出非合法 JSON：{raw_text[:80]}")

    intent = parsed.get("intent", "")
    # 校验 intent 是否合法
    if intent not in (INTENT_KNOWLEDGE, INTENT_QUERY, INTENT_OPERATION):
        print(f"  ⚠️  [json_parse] 未知意图类型 [{intent}]，降级为知识型")
        intent = FALLBACK_INTENT

    return {
        "intent":   intent,
        "reason":   parsed.get("reason", "（模型未提供原因）"),
        "params":   parsed.get("params", {}),
        "mode":     "json_parse",
        "fallback": False,
    }


# ══════════════════════════════════════════════════════════════
# 公共工具函数
# ══════════════════════════════════════════════════════════════

def _try_parse_json(text: str) -> Optional[dict]:
    """
    多策略解析 JSON，依次尝试：
    1. 直接解析（最常见情况）
    2. 提取 markdown 代码块内的 JSON（模型偶尔会加 ```json ... ```）
    3. 提取文本中最外层的花括号内容

    返回解析成功的 dict，或 None（所有策略均失败）。
    """
    text = text.strip()

    # 策略1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2：从 markdown 代码块提取
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略3：提取最外层花括号
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _fallback_result(reason: str) -> dict:
    """生成兜底路由结果，默认走知识型 RAG，并标记 fallback=True。"""
    print(f"  ⚠️  路由异常，降级为知识型 RAG。原因：{reason}")
    return {
        "intent":   FALLBACK_INTENT,
        "reason":   f"[兜底] {reason}",
        "params":   {},
        "mode":     ROUTING_MODE,
        "fallback": True,
    }


# ══════════════════════════════════════════════════════════════
# 对外入口
# ══════════════════════════════════════════════════════════════

def route(question: str) -> dict:
    """
    对外统一入口：根据 ROUTING_MODE 选择路由方式，返回意图判断结果。

    用法：
        from router import route
        result = route("帮我查一下C002客户的账户情况")
        # result = {"intent": "查询型", "reason": "...", "params": {"customer_id": "C002"}, ...}

    可通过以下方式切换路由模式（重启生效）：
        export ROUTING_MODE=tool_calling   # 或
        export ROUTING_MODE=json_parse
    """
    if ROUTING_MODE == "tool_calling":
        return _route_tool_calling(question)
    else:
        return _route_json_parse(question)
