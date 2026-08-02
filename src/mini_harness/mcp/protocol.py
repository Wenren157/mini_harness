import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


# ============================================================
# JSON-RPC 消息结构
# ============================================================
@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JSONRPCResponse:
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# ============================================================
# MCP 错误码（标准）
# ============================================================
class MCPErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_FAILED = -32002


def make_error_response(id: Optional[int], code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    }


def make_success_response(id: Optional[int], result: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }


# TODO: 未来可扩展支持 resources/list, prompts/get, sampling/createMessage