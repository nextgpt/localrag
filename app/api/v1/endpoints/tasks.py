"""
任务管理API端点
处理异步任务的查询、管理和监控
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

from app.models.responses import SuccessResponse, PaginatedResponse, PaginationInfo, ErrorCode
from app.models.requests import TaskQueryRequest, TaskStatus, TaskManagementRequest
from app.core.exceptions import create_task_exception, create_file_exception
from app.services.task_service import get_task_manager, TaskManager
from app.services.document_service import get_document_service, DocumentService
from app.services.cache_service import get_cache_service, CacheService

router = APIRouter(prefix="/tasks", tags=["任务管理"])
logger = logging.getLogger("rag-anything")


@router.get("/{task_id}", response_model=SuccessResponse, summary="获取任务状态")
async def get_task_status(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    获取任务状态接口
    
    **功能说明:**
    - 获取指定任务的详细状态信息
    - 包括任务进度、结果、错误信息等
    - 支持实时状态查询
    
    **路径参数:**
    - task_id: 任务ID（必填）
    
    **响应数据:**
    - task_id: 任务ID
    - task_name: 任务名称
    - status: 任务状态（pending/running/completed/failed）
    - progress: 任务进度（0-100）
    - created_at: 创建时间
    - started_at: 开始时间
    - completed_at: 完成时间
    - result: 任务结果（如果已完成）
    - error: 错误信息（如果失败）
    """
    
    try:
        task_status = await task_manager.get_task_status(task_id)
        
        if task_status is None:
            raise create_task_exception(
                ErrorCode.TASK_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        return SuccessResponse(
            data=task_status,
            message="获取任务状态成功"
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {task_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_task_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取任务状态失败: {str(e)}"
            )


@router.get("/{task_id}/result", response_model=SuccessResponse, summary="获取任务结果")
async def get_task_result(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    获取任务结果接口
    
    **功能说明:**
    - 获取已完成任务的执行结果
    - 只有状态为completed的任务才有结果
    - 如果任务失败或正在运行，会返回相应错误
    
    **路径参数:**
    - task_id: 任务ID（必填）
    
    **响应数据:**
    - task_id: 任务ID
    - status: 任务状态
    - result: 任务执行结果
    - completed_at: 完成时间
    """
    
    try:
        result = await task_manager.get_task_result(task_id)
        
        # 获取任务状态信息
        task_status = await task_manager.get_task_status(task_id)
        
        result_data = {
            "task_id": task_id,
            "status": task_status.get("status"),
            "result": result,
            "completed_at": task_status.get("completed_at")
        }
        
        return SuccessResponse(
            data=result_data,
            message="获取任务结果成功"
        )
        
    except Exception as e:
        logger.error(f"获取任务结果失败: {task_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_task_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取任务结果失败: {str(e)}"
            )


@router.post("/{task_id}/cancel", response_model=SuccessResponse, summary="取消任务")
async def cancel_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    取消任务接口
    
    **功能说明:**
    - 取消正在执行或等待中的任务
    - 已完成或已失败的任务无法取消
    - 取消操作是异步的，可能需要一定时间
    
    **路径参数:**
    - task_id: 任务ID（必填）
    
    **响应数据:**
    - task_id: 任务ID
    - cancelled: 是否成功取消
    - previous_status: 取消前的状态
    - cancelled_at: 取消时间
    """
    
    try:
        # 获取取消前的状态
        task_status = await task_manager.get_task_status(task_id)
        if task_status is None:
            raise create_task_exception(
                ErrorCode.TASK_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        previous_status = task_status.get("status")
        
        # 取消任务
        cancelled = await task_manager.cancel_task(task_id)
        
        result_data = {
            "task_id": task_id,
            "cancelled": cancelled,
            "previous_status": previous_status,
            "cancelled_at": task_status.get("completed_at") if cancelled else None
        }
        
        message = "任务取消成功" if cancelled else "任务无法取消（可能已完成）"
        
        return SuccessResponse(
            data=result_data,
            message=message
        )
        
    except Exception as e:
        logger.error(f"取消任务失败: {task_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_task_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"取消任务失败: {str(e)}"
            )


@router.get("/", response_model=PaginatedResponse, summary="列出任务")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    created_by: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    列出任务接口
    
    **功能说明:**
    - 获取系统中的任务列表
    - 支持按状态和创建者过滤
    - 支持分页查询
    - 按创建时间倒序排列
    
    **查询参数:**
    - status: 任务状态过滤（可选）
    - created_by: 创建者过滤（可选）
    - limit: 每页数量（默认10，最大100）
    - offset: 偏移量（默认0）
    
    **响应数据:**
    - tasks: 任务列表
    - pagination: 分页信息
    - filter_info: 过滤条件信息
    """
    
    if limit <= 0 or limit > 100:
        raise create_task_exception(
            ErrorCode.INVALID_REQUEST,
            "每页数量必须在1-100之间"
        )
    
    try:
        tasks = await task_manager.list_tasks(
            status_filter=status,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
        
        # 计算总数（这里简化了，实际应该有专门的计数方法）
        total_count = len(tasks) if len(tasks) < limit else limit + offset + 1
        
        # 计算分页信息
        pages = (total_count + limit - 1) // limit
        pagination = PaginationInfo(
            page=(offset // limit) + 1,
            size=limit,
            total=total_count,
            pages=pages
        )
        
        # 过滤条件信息
        filter_info = {
            "status_filter": status,
            "created_by_filter": created_by,
            "applied_filters": []
        }
        
        if status:
            filter_info["applied_filters"].append(f"status={status}")
        if created_by:
            filter_info["applied_filters"].append(f"created_by={created_by}")
        
        result_data = {
            "tasks": tasks,
            "filter_info": filter_info
        }
        
        return PaginatedResponse(
            data=result_data,
            pagination=pagination,
            message=f"获取任务列表成功，共{len(tasks)}个任务"
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise create_task_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取任务列表失败: {str(e)}"
        )


@router.get("/stats/summary", response_model=SuccessResponse, summary="获取任务统计信息")
async def get_task_stats(
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    获取任务统计信息接口
    
    **功能说明:**
    - 获取系统的任务统计信息
    - 包括各状态任务数量
    - 提供系统负载和性能指标
    
    **响应数据:**
    - total: 总任务数
    - pending: 等待中任务数
    - running: 运行中任务数
    - completed: 已完成任务数
    - failed: 失败任务数
    - success_rate: 成功率
    """
    
    try:
        stats = await task_manager.get_task_count()
        
        # 计算成功率
        total_finished = stats.get("completed", 0) + stats.get("failed", 0)
        success_rate = (stats.get("completed", 0) / total_finished * 100) if total_finished > 0 else 0
        
        result_data = {
            **stats,
            "success_rate": round(success_rate, 2)
        }
        
        return SuccessResponse(
            data=result_data,
            message="获取任务统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取任务统计信息失败: {e}")
        raise create_task_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取任务统计信息失败: {str(e)}"
        )


@router.delete("/{task_id}", response_model=SuccessResponse, summary="删除任务记录")
async def delete_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    删除任务记录接口
    
    **功能说明:**
    - 删除指定的任务记录
    - 如果任务正在运行，会先取消然后删除
    - 删除后无法恢复任务信息
    
    **路径参数:**
    - task_id: 任务ID（必填）
    
    **响应数据:**
    - task_id: 任务ID
    - deleted: 是否成功删除
    - was_running: 删除时是否正在运行
    """
    
    try:
        # 检查任务是否存在
        task_status = await task_manager.get_task_status(task_id)
        if task_status is None:
            raise create_task_exception(
                ErrorCode.TASK_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        was_running = task_status.get("status") == TaskStatus.RUNNING
        
        # 删除任务
        deleted = await task_manager.remove_task(task_id)
        
        result_data = {
            "task_id": task_id,
            "deleted": deleted,
            "was_running": was_running
        }
        
        return SuccessResponse(
            data=result_data,
            message="任务记录删除成功"
        )
        
    except Exception as e:
        logger.error(f"删除任务记录失败: {task_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_task_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"删除任务记录失败: {str(e)}"
            )


@router.get("/queue/stats", response_model=SuccessResponse, summary="获取任务队列统计")
async def get_queue_stats(
    queue_name: str = Query("document_parse", description="队列名称"),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    获取任务队列统计信息 - 类似mineru-web的任务监控
    
    **功能说明:**
    - 实时队列状态监控
    - 任务执行统计
    - 性能指标分析
    
    **查询参数:**
    - queue_name: 队列名称（document_parse/document_vectorize）
    
    **响应数据:**
    - pending_tasks: 待处理任务数
    - priority_tasks: 优先级任务数
    - running_tasks: 运行中任务数
    - completed_tasks: 已完成任务数
    - failed_tasks: 失败任务数
    """
    try:
        stats = await cache_service.get_queue_stats(queue_name)
        
        return SuccessResponse(
            data={
                "queue_name": queue_name,
                "statistics": stats,
                "updated_at": datetime.utcnow().isoformat()
            },
            message="队列统计获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取队列统计失败: {queue_name} - {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取队列统计失败: {str(e)}"
        )


@router.get("/processing/overview", response_model=SuccessResponse, summary="获取处理概览")
async def get_processing_overview(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取系统处理概览 - 参考mineru-web的仪表板
    
    **功能说明:**
    - 系统整体处理状态
    - 文件处理统计
    - 性能指标监控
    - 系统健康状态
    
    **响应数据:**
    - file_statistics: 文件统计信息
    - queue_statistics: 队列统计信息  
    - performance_metrics: 性能指标
    - system_health: 系统健康状态
    """
    try:
        statistics = await document_service.get_processing_statistics()
        
        return SuccessResponse(
            data=statistics,
            message="处理概览获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取处理概览失败: {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取处理概览失败: {str(e)}"
        )


@router.get("/retry-candidates", response_model=SuccessResponse, summary="获取重试候选任务")
async def get_retry_candidates(
    queue_name: str = Query("document_parse", description="队列名称"),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    获取需要重试的任务列表 - 参考mineru-web的错误恢复机制
    
    **功能说明:**
    - 自动识别失败任务
    - 智能重试调度
    - 错误原因分析
    
    **查询参数:**
    - queue_name: 队列名称
    
    **响应数据:**
    - retry_tasks: 可重试任务列表
    - total_count: 总任务数
    - retry_ready: 准备重试的任务数
    """
    try:
        retry_tasks = await cache_service.get_failed_tasks_for_retry(queue_name)
        
        retry_ready = [
            task for task in retry_tasks 
            if task["current_retries"] < task["max_retries"]
        ]
        
        result_data = {
            "queue_name": queue_name,
            "retry_tasks": retry_tasks,
            "total_count": len(retry_tasks),
            "retry_ready": len(retry_ready),
            "checked_at": datetime.utcnow().isoformat()
        }
        
        return SuccessResponse(
            data=result_data,
            message="重试候选任务获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取重试候选任务失败: {queue_name} - {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取重试候选任务失败: {str(e)}"
        )


@router.post("/manage", response_model=SuccessResponse, summary="批量任务管理")
async def manage_tasks(
    request: TaskManagementRequest,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    批量任务管理操作 - 参考mineru-web的任务控制功能
    
    **功能说明:**
    - 批量取消任务
    - 批量重试任务
    - 调整任务优先级
    - 暂停/恢复任务
    
    **请求体:**
    - action: 操作类型（cancel/retry/priority/pause/resume）
    - task_ids: 任务ID列表
    - options: 操作选项
    
    **响应数据:**
    - action: 执行的操作
    - processed_tasks: 处理的任务数
    - successful_operations: 成功操作数
    - failed_operations: 失败操作数
    - results: 详细结果
    """
    try:
        action = request.action
        task_ids = request.task_ids
        options = request.options or {}
        
        if not task_ids:
            raise create_file_exception(
                ErrorCode.INVALID_REQUEST,
                "任务ID列表不能为空"
            )
        
        results = []
        successful_operations = 0
        failed_operations = 0
        
        for task_id in task_ids:
            try:
                if action == "cancel":
                    # 取消任务
                    await cache_service.hset(f"task:{task_id}", "status", "cancelled")
                    await cache_service.hset(f"task:{task_id}", "cancelled_at", datetime.utcnow().isoformat())
                    
                    results.append({
                        "task_id": task_id,
                        "action": "cancel",
                        "success": True,
                        "message": "任务已取消"
                    })
                    
                elif action == "retry":
                    # 设置任务重试
                    max_retries = options.get("max_retries", 3)
                    delay = options.get("delay", 60)
                    
                    success = await cache_service.setup_task_retry(task_id, max_retries, delay)
                    
                    if success:
                        await cache_service.hset(f"task:{task_id}", "status", "pending_retry")
                    
                    results.append({
                        "task_id": task_id,
                        "action": "retry",
                        "success": success,
                        "message": "重试设置成功" if success else "重试设置失败"
                    })
                    
                elif action == "priority":
                    # 调整优先级
                    new_priority = options.get("priority", 1)
                    await cache_service.hset(f"task:{task_id}", "priority", new_priority)
                    
                    results.append({
                        "task_id": task_id,
                        "action": "priority",
                        "success": True,
                        "message": f"优先级已调整为 {new_priority}"
                    })
                    
                elif action == "pause":
                    # 暂停任务
                    await cache_service.hset(f"task:{task_id}", "status", "paused")
                    await cache_service.hset(f"task:{task_id}", "paused_at", datetime.utcnow().isoformat())
                    
                    results.append({
                        "task_id": task_id,
                        "action": "pause",
                        "success": True,
                        "message": "任务已暂停"
                    })
                    
                elif action == "resume":
                    # 恢复任务
                    await cache_service.hset(f"task:{task_id}", "status", "pending")
                    await cache_service.hdel(f"task:{task_id}", "paused_at")
                    
                    results.append({
                        "task_id": task_id,
                        "action": "resume",
                        "success": True,
                        "message": "任务已恢复"
                    })
                    
                else:
                    results.append({
                        "task_id": task_id,
                        "action": action,
                        "success": False,
                        "error": f"不支持的操作类型: {action}"
                    })
                    failed_operations += 1
                    continue
                
                successful_operations += 1
                
            except Exception as e:
                logger.error(f"任务管理操作失败: {task_id} - {action} - {e}")
                results.append({
                    "task_id": task_id,
                    "action": action,
                    "success": False,
                    "error": str(e)
                })
                failed_operations += 1
        
        result_data = {
            "action": action,
            "processed_tasks": len(task_ids),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "results": results,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"批量任务管理完成: {action} - 成功{successful_operations}个，失败{failed_operations}个")
        
        return SuccessResponse(
            data=result_data,
            message=f"批量{action}操作完成"
        )
        
    except Exception as e:
        logger.error(f"批量任务管理失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"任务管理操作失败: {str(e)}"
            )


@router.get("/{task_id}/details", response_model=SuccessResponse, summary="获取任务详细信息")
async def get_task_details(
    task_id: str,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    获取任务详细信息 - 类似mineru-web的任务监控面板
    
    **功能说明:**
    - 任务完整状态信息
    - 执行进度跟踪
    - 错误信息分析
    - 性能指标统计
    
    **路径参数:**
    - task_id: 任务ID
    
    **响应数据:**
    - task_info: 任务基本信息
    - execution_details: 执行详细信息
    - performance_metrics: 性能指标
    - error_info: 错误信息（如有）
    """
    try:
        # 获取任务基本信息
        task_info = await cache_service.get_task_info(task_id)
        if not task_info:
            raise create_file_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        # 获取任务执行详情
        task_hash = f"task:{task_id}"
        execution_details = await cache_service.hgetall(task_hash)
        
        # 获取重试信息（如果有）
        retry_info = await cache_service.hgetall(f"task:{task_id}:retry")
        
        # 计算执行时间
        performance_metrics = {}
        if execution_details.get("started_at") and execution_details.get("completed_at"):
            start_time = datetime.fromisoformat(execution_details["started_at"])
            end_time = datetime.fromisoformat(execution_details["completed_at"])
            execution_time = (end_time - start_time).total_seconds()
            performance_metrics["execution_time_seconds"] = execution_time
        
        result_data = {
            "task_id": task_id,
            "task_info": task_info,
            "execution_details": execution_details,
            "retry_info": retry_info if retry_info else None,
            "performance_metrics": performance_metrics,
            "error_info": execution_details.get("error") if execution_details.get("status") == "failed" else None,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
        return SuccessResponse(
            data=result_data,
            message="任务详细信息获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取任务详细信息失败: {task_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取任务详情失败: {str(e)}"
            )


@router.get("/{file_id}/processing-status", response_model=SuccessResponse, summary="获取文件处理状态")
async def get_file_processing_status(
    file_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文件完整处理状态 - 类似mineru-web的文件状态监控
    
    **功能说明:**
    - 文件处理进度跟踪
    - 解析和向量化状态
    - 任务依赖关系显示
    - 错误诊断信息
    
    **路径参数:**
    - file_id: 文件ID
    
    **响应数据:**
    - file_info: 文件基本信息
    - total_progress: 总体进度百分比
    - parse: 解析状态和详情
    - vectorize: 向量化状态和详情
    - last_updated: 最后更新时间
    """
    try:
        status_data = await document_service.get_file_processing_status(file_id)
        
        if not status_data.get("exists"):
            raise create_file_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件不存在: {file_id}"
            )
        
        return SuccessResponse(
            data=status_data,
            message="文件处理状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取文件处理状态失败: {file_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取文件处理状态失败: {str(e)}"
            ) 