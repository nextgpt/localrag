"""
异常处理系统
统一处理各种错误情况，确保返回一致的错误响应格式
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import traceback
import uuid
import logging
from datetime import datetime
from typing import Union

from app.models.responses import ErrorResponse, ErrorDetail, ErrorCode

# 设置日志记录器
logger = logging.getLogger("rag-anything")


class RAGException(Exception):
    """RAG系统基础异常类"""
    
    def __init__(self, code: ErrorCode, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FileException(RAGException):
    """文件处理相关异常"""
    pass


class TaskException(RAGException):
    """任务处理相关异常"""
    pass


class SearchException(RAGException):
    """检索相关异常"""
    pass


class ServiceException(RAGException):
    """外部服务连接异常"""
    pass


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """RAG系统自定义异常处理器"""
    request_id = str(uuid.uuid4())
    
    # 记录错误日志
    logger.error(f"RAG异常 [{request_id}]: {exc.code} - {exc.message}", extra={
        "request_id": request_id,
        "path": str(request.url.path),
        "method": request.method,
        "details": exc.details
    })
    
    # 根据错误类型确定HTTP状态码
    status_code_map = {
        ErrorCode.NOT_FOUND: 404,
        ErrorCode.FILE_NOT_FOUND: 404,
        ErrorCode.TASK_NOT_FOUND: 404,
        ErrorCode.UNAUTHORIZED: 401,
        ErrorCode.FORBIDDEN: 403,
        ErrorCode.INVALID_REQUEST: 400,
        ErrorCode.INVALID_FILE_TYPE: 400,
        ErrorCode.FILE_TOO_LARGE: 413,
        ErrorCode.RATE_LIMIT_EXCEEDED: 429,
        ErrorCode.REQUEST_TIMEOUT: 408,
        ErrorCode.TASK_TIMEOUT: 408,
    }
    
    status_code = status_code_map.get(exc.code, 500)
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details if exc.details else None
        ),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


async def http_exception_handler_custom(request: Request, exc: HTTPException) -> JSONResponse:
    """自定义HTTP异常处理器"""
    request_id = str(uuid.uuid4())
    
    # 映射HTTP状态码到错误代码
    code_map = {
        400: ErrorCode.INVALID_REQUEST,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        405: ErrorCode.METHOD_NOT_ALLOWED,
        408: ErrorCode.REQUEST_TIMEOUT,
        413: ErrorCode.FILE_TOO_LARGE,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_SERVER_ERROR,
    }
    
    error_code = code_map.get(exc.status_code, ErrorCode.INTERNAL_SERVER_ERROR)
    
    logger.warning(f"HTTP异常 [{request_id}]: {exc.status_code} - {exc.detail}", extra={
        "request_id": request_id,
        "path": str(request.url.path),
        "method": request.method,
        "status_code": exc.status_code
    })
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=error_code,
            message=str(exc.detail)
        ),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理器 - 处理所有未捕获的异常"""
    request_id = str(uuid.uuid4())
    
    # 记录详细的错误信息
    error_traceback = traceback.format_exc()
    logger.error(f"未处理异常 [{request_id}]: {str(exc)}", extra={
        "request_id": request_id,
        "path": str(request.url.path),
        "method": request.method,
        "traceback": error_traceback,
        "exception_type": type(exc).__name__
    })
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="服务器内部错误，请稍后重试"
        ),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    """Pydantic验证异常处理器"""
    request_id = str(uuid.uuid4())
    
    # 提取验证错误详情
    errors = []
    for error in exc.errors():
        field = ".".join([str(x) for x in error["loc"]])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(f"参数验证失败 [{request_id}]: {errors}", extra={
        "request_id": request_id,
        "path": str(request.url.path),
        "method": request.method,
        "validation_errors": errors
    })
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=ErrorCode.INVALID_REQUEST,
            message="请求参数验证失败",
            details={"validation_errors": errors}
        ),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """设置异常处理器"""
    
    # 导入Pydantic验证异常
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError
    
    # 注册异常处理器
    app.exception_handler(RAGException)(rag_exception_handler)
    app.exception_handler(HTTPException)(http_exception_handler_custom)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(ValidationError)(validation_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)
    
    logger.info("异常处理器设置完成")


# 便捷的异常创建函数
def create_file_exception(code: ErrorCode, message: str, details: dict = None) -> FileException:
    """创建文件异常"""
    return FileException(code, message, details)


def create_task_exception(code: ErrorCode, message: str, details: dict = None) -> TaskException:
    """创建任务异常"""
    return TaskException(code, message, details)


def create_search_exception(code: ErrorCode, message: str, details: dict = None) -> SearchException:
    """创建检索异常"""
    return SearchException(code, message, details)


def create_service_exception(code: ErrorCode, message: str, details: dict = None) -> ServiceException:
    """创建服务异常"""
    return ServiceException(code, message, details) 