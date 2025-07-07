"""
系统健康检查API端点
监控系统各组件的运行状态
"""

from fastapi import APIRouter, Depends
import asyncio
import httpx
import logging
from datetime import datetime
from typing import Dict, Any
import psutil

from app.models.responses import SuccessResponse, ErrorCode, HealthCheckResponse, ErrorResponse
from app.core.config import settings
from app.core.exceptions import create_service_exception
from app.services import get_service_status

router = APIRouter(prefix="/health", tags=["系统健康"])
logger = logging.getLogger("rag-anything")


async def check_qdrant_health() -> str:
    """检查Qdrant向量数据库健康状态"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/")
            if response.status_code == 200:
                return "healthy"
            else:
                return f"unhealthy (status: {response.status_code})"
    except Exception as e:
        return f"unreachable ({str(e)})"


async def check_redis_health() -> str:
    """检查Redis数据库健康状态"""
    try:
        # 这里需要实际的Redis连接检查
        # 目前返回模拟状态
        import redis.asyncio as redis
        
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            socket_connect_timeout=5
        )
        
        await redis_client.ping()
        await redis_client.close()
        return "healthy"
        
    except Exception as e:
        return f"unreachable ({str(e)})"


async def check_minio_health() -> str:
    """检查MinIO对象存储健康状态"""
    try:
        protocol = "https" if settings.MINIO_SECURE else "http"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{protocol}://{settings.MINIO_HOST}:{settings.MINIO_PORT}/minio/health/live")
            if response.status_code == 200:
                return "healthy"
            else:
                return f"unhealthy (status: {response.status_code})"
    except Exception as e:
        return f"unreachable ({str(e)})"


async def check_sglang_health() -> str:
    """检查SGLang服务健康状态"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.SGLANG_BASE_URL}/health")
            if response.status_code == 200:
                return "healthy"
            else:
                # 尝试其他可能的健康检查端点
                response = await client.get(f"{settings.SGLANG_BASE_URL}/v1/models")
                if response.status_code == 200:
                    return "healthy"
                else:
                    return f"unhealthy (status: {response.status_code})"
    except Exception as e:
        return f"unreachable ({str(e)})"


async def check_embedding_health() -> str:
    """检查嵌入服务健康状态"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.EMBEDDING_BASE_URL}/models")
            if response.status_code == 200:
                data = response.json()
                # 检查模型列表中是否包含我们需要的模型
                if "data" in data:
                    model_ids = [model.get("id", "") for model in data["data"]]
                    if settings.EMBEDDING_MODEL in model_ids:
                        return "healthy"
                    else:
                        return f"model_not_found ({settings.EMBEDDING_MODEL})"
                return "healthy"
            else:
                return f"unhealthy (status: {response.status_code})"
    except Exception as e:
        return f"unreachable ({str(e)})"


@router.get("/", response_model=SuccessResponse, summary="系统健康检查")
async def health_check():
    """
    系统健康检查接口
    
    **功能说明:**
    - 检查所有核心服务的运行状态
    - 包括向量数据库、缓存、对象存储、LLM服务等
    - 提供详细的健康状态报告
    - 用于系统监控和故障诊断
    
    **响应数据:**
    - server: 服务器状态
    - qdrant: Qdrant向量数据库状态
    - redis: Redis缓存数据库状态
    - minio: MinIO对象存储状态
    - sglang: SGLang LLM服务状态
    - embedding: 嵌入服务状态
    - overall_status: 总体健康状态
    - check_time: 检查时间
    """
    
    start_time = datetime.utcnow()
    
    try:
        # 并行检查所有服务
        qdrant_status, redis_status, minio_status, sglang_status, embedding_status = await asyncio.gather(
            check_qdrant_health(),
            check_redis_health(),
            check_minio_health(),
            check_sglang_health(),
            check_embedding_health(),
            return_exceptions=True
        )
        
        # 处理异常情况
        def format_status(status):
            if isinstance(status, Exception):
                return f"error ({str(status)})"
            return status
        
        qdrant_status = format_status(qdrant_status)
        redis_status = format_status(redis_status)
        minio_status = format_status(minio_status)
        sglang_status = format_status(sglang_status)
        embedding_status = format_status(embedding_status)
        
        # 计算总体健康状态
        all_statuses = [qdrant_status, redis_status, minio_status, sglang_status, embedding_status]
        healthy_count = sum(1 for status in all_statuses if status == "healthy")
        total_services = len(all_statuses)
        
        if healthy_count == total_services:
            overall_status = "healthy"
        elif healthy_count >= total_services * 0.6:  # 60%以上服务正常
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        end_time = datetime.utcnow()
        check_duration = (end_time - start_time).total_seconds()
        
        health_data = {
            "server": "healthy",
            "qdrant": qdrant_status,
            "redis": redis_status,
            "minio": minio_status,
            "sglang": sglang_status,
            "embedding": embedding_status,
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_services,
            "check_time": end_time.isoformat(),
            "check_duration_seconds": round(check_duration, 3)
        }
        
        logger.info(f"健康检查完成: {overall_status} ({healthy_count}/{total_services} 服务正常)")
        
        return SuccessResponse(
            data=health_data,
            message=f"系统健康检查完成 - {overall_status}"
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return SuccessResponse(
            data={
                "server": "healthy",
                "qdrant": "unknown",
                "redis": "unknown",
                "minio": "unknown",
                "sglang": "unknown",
                "embedding": "unknown",
                "overall_status": "error",
                "error": str(e),
                "check_time": datetime.utcnow().isoformat()
            },
            message="健康检查过程中发生错误"
        )


@router.get("/quick", response_model=SuccessResponse, summary="快速健康检查")
async def quick_health_check():
    """
    快速健康检查接口
    
    **功能说明:**
    - 快速检查服务器基本状态
    - 不检查外部依赖服务
    - 响应时间更快，适合频繁调用
    
    **响应数据:**
    - status: 服务器状态
    - timestamp: 检查时间
    - uptime: 服务运行时间（如果可获取）
    """
    
    try:
        # 获取系统基本信息
        import psutil
        import os
        
        # CPU和内存使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 进程信息
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "process_memory_mb": round(process_memory.rss / 1024 / 1024, 2)
            },
            "settings": {
                "max_concurrent_files": settings.MAX_CONCURRENT_FILES,
                "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
                "allowed_extensions": len(settings.ALLOWED_EXTENSIONS)
            }
        }
        
        return SuccessResponse(
            data=health_data,
            message="快速健康检查完成"
        )
        
    except Exception as e:
        logger.warning(f"快速健康检查失败: {e}")
        return SuccessResponse(
            data={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "note": "系统信息获取失败，但服务正常运行"
            },
            message="快速健康检查完成（部分信息不可用）"
        )


@router.get("/services/{service_name}", response_model=SuccessResponse, summary="单个服务健康检查")
async def check_single_service(service_name: str):
    """
    单个服务健康检查接口
    
    **功能说明:**
    - 检查指定服务的健康状态
    - 提供更详细的服务信息
    - 支持的服务：qdrant、redis、minio、sglang、embedding
    
    **路径参数:**
    - service_name: 服务名称
    
    **响应数据:**
    - service: 服务名称
    - status: 健康状态
    - details: 详细信息
    - check_time: 检查时间
    """
    
    service_checkers = {
        "qdrant": check_qdrant_health,
        "redis": check_redis_health,
        "minio": check_minio_health,
        "sglang": check_sglang_health,
        "embedding": check_embedding_health
    }
    
    if service_name not in service_checkers:
        raise create_service_exception(
            ErrorCode.INVALID_REQUEST,
            f"不支持的服务名称: {service_name}，支持的服务: {', '.join(service_checkers.keys())}"
        )
    
    try:
        start_time = datetime.utcnow()
        status = await service_checkers[service_name]()
        end_time = datetime.utcnow()
        
        check_duration = (end_time - start_time).total_seconds()
        
        # 根据服务提供额外的详细信息
        details = {"response_time_seconds": round(check_duration, 3)}
        
        if service_name == "qdrant":
            details["host"] = f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        elif service_name == "redis":
            details["host"] = f"{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        elif service_name == "minio":
            details["host"] = f"{settings.MINIO_HOST}:{settings.MINIO_PORT}"
        elif service_name == "sglang":
            details["base_url"] = settings.SGLANG_BASE_URL
        elif service_name == "embedding":
            details["base_url"] = settings.EMBEDDING_BASE_URL
            details["model"] = settings.EMBEDDING_MODEL
        
        result_data = {
            "service": service_name,
            "status": status,
            "details": details,
            "check_time": end_time.isoformat()
        }
        
        return SuccessResponse(
            data=result_data,
            message=f"{service_name} 服务检查完成"
        )
        
    except Exception as e:
        logger.error(f"检查服务 {service_name} 失败: {e}")
        return SuccessResponse(
            data={
                "service": service_name,
                "status": f"error ({str(e)})",
                "check_time": datetime.utcnow().isoformat()
            },
            message=f"{service_name} 服务检查失败"
        )


@router.get("/services", response_model=SuccessResponse)
async def service_status():
    """
    服务状态详情
    
    返回各个服务的详细状态信息
    """
    try:
        service_status = await get_service_status()
        
        return SuccessResponse(
            data=service_status,
            message="服务状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取服务状态失败: {e}")
        return ErrorResponse(
            message="获取服务状态失败",
            error_code="SERVICE_STATUS_FAILED",
            data={"error": str(e)}
        )


@router.get("/system", response_model=SuccessResponse)
async def system_status():
    """
    系统资源状态
    
    返回系统CPU、内存、磁盘等资源使用情况
    """
    try:
        system_info = _get_system_info()
        
        return SuccessResponse(
            data=system_info,
            message="系统状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return ErrorResponse(
            message="获取系统状态失败",
            error_code="SYSTEM_STATUS_FAILED",
            data={"error": str(e)}
        )


def _get_system_info() -> Dict[str, Any]:
    """获取系统资源信息"""
    try:
        # CPU 信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        }
        
        # 网络信息
        network = psutil.net_io_counters()
        network_info = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        return {
            "error": str(e),
            "status": "unavailable"
        }


def _calculate_overall_status(service_status: Dict[str, Any]) -> str:
    """计算整体健康状态"""
    if not service_status.get("initialized", False):
        return "initializing"
    
    services = service_status.get("services", {})
    if not services:
        return "unknown"
    
    # 检查各个服务状态
    critical_services = ["minio", "cache", "vector"]  # 关键服务
    optional_services = ["task", "document", "search"]  # 可选服务
    
    critical_healthy = 0
    critical_total = 0
    optional_healthy = 0
    optional_total = 0
    
    for service_name, service_info in services.items():
        status = service_info.get("status", "unknown")
        
        if service_name in critical_services:
            critical_total += 1
            if status == "healthy":
                critical_healthy += 1
        elif service_name in optional_services:
            optional_total += 1
            if status == "healthy":
                optional_healthy += 1
    
    # 计算健康状态
    if critical_total > 0 and critical_healthy == critical_total:
        if optional_total == 0 or optional_healthy == optional_total:
            return "healthy"
        elif optional_healthy >= optional_total * 0.5:
            return "degraded"
        else:
            return "unhealthy"
    elif critical_healthy >= critical_total * 0.5:
        return "degraded"
    else:
        return "unhealthy"


@router.get("/readiness")
async def readiness_check():
    """
    就绪性检查
    
    检查服务是否准备好接收请求
    主要用于容器编排系统（如 Kubernetes）
    """
    try:
        service_status = await get_service_status()
        
        if not service_status.get("initialized", False):
            return {"status": "not_ready", "message": "服务尚未初始化完成"}
        
        # 检查关键服务是否健康
        services = service_status.get("services", {})
        critical_services = ["minio", "cache", "vector"]
        
        for service_name in critical_services:
            service_info = services.get(service_name, {})
            if service_info.get("status") != "healthy":
                return {
                    "status": "not_ready", 
                    "message": f"关键服务 {service_name} 未就绪"
                }
        
        return {"status": "ready", "message": "服务已就绪"}
        
    except Exception as e:
        logger.error(f"就绪性检查失败: {e}")
        return {"status": "not_ready", "message": f"检查失败: {str(e)}"}


@router.get("/liveness")
async def liveness_check():
    """
    存活性检查
    
    检查服务是否还在运行
    主要用于容器编排系统（如 Kubernetes）
    """
    try:
        # 简单的存活性检查 - 检查能否正常响应
        current_time = datetime.now().isoformat()
        
        return {
            "status": "alive",
            "timestamp": current_time,
            "message": "服务正在运行"
        }
        
    except Exception as e:
        logger.error(f"存活性检查失败: {e}")
        return {"status": "dead", "message": f"检查失败: {str(e)}"} 