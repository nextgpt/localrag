"""
API 响应数据模型
定义统一的成功和错误响应格式
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class ErrorCode(str, Enum):
    """错误代码枚举 - 符合国际标准"""
    
    # === 通用错误 ===
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # === 文件处理错误 ===
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FILE_UPLOAD_FAILED = "FILE_UPLOAD_FAILED"
    FILE_DOWNLOAD_FAILED = "FILE_DOWNLOAD_FAILED"
    FILE_PARSE_FAILED = "FILE_PARSE_FAILED"
    FILE_DELETE_FAILED = "FILE_DELETE_FAILED"
    
    # === 任务处理错误 ===
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_CREATION_FAILED = "TASK_CREATION_FAILED"
    TASK_ALREADY_RUNNING = "TASK_ALREADY_RUNNING"
    TASK_FAILED = "TASK_FAILED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    
    # === 检索错误 ===
    SEARCH_FAILED = "SEARCH_FAILED"
    INVALID_SEARCH_PARAMS = "INVALID_SEARCH_PARAMS"
    SEARCH_TIMEOUT = "SEARCH_TIMEOUT"
    
    # === 服务连接错误 ===
    QDRANT_CONNECTION_ERROR = "QDRANT_CONNECTION_ERROR"
    REDIS_CONNECTION_ERROR = "REDIS_CONNECTION_ERROR"
    MINIO_CONNECTION_ERROR = "MINIO_CONNECTION_ERROR"
    SGLANG_CONNECTION_ERROR = "SGLANG_CONNECTION_ERROR"
    EMBEDDING_CONNECTION_ERROR = "EMBEDDING_CONNECTION_ERROR"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"


class ErrorDetail(BaseModel):
    """错误详情模型"""
    code: ErrorCode
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="响应时间戳"
    )
    request_id: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseResponse):
    """成功响应模型"""
    success: bool = True
    data: Any
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error: ErrorDetail
    data: Optional[Any] = None  # 添加可选的data字段以兼容SuccessResponse


class PaginationInfo(BaseModel):
    """分页信息模型"""
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页大小")
    total: int = Field(..., description="总记录数")
    pages: int = Field(..., description="总页数")


class PaginatedResponse(BaseModel):
    """分页响应模型"""
    success: bool = Field(True, description="操作是否成功")
    data: Any = Field(..., description="响应数据")
    pagination: PaginationInfo = Field(..., description="分页信息")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="响应时间")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="健康状态")
    services: Dict[str, str] = Field(..., description="各服务状态")
    timestamp: str = Field(..., description="检查时间") 