"""
文档管理API端点
处理文档的删除、解析、索引等操作
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
import logging
from datetime import datetime, timedelta

from app.models.responses import SuccessResponse, ErrorCode
from app.models.requests import FileDeleteRequest, DocumentProcessRequest, VectorStoreRequest, BatchFileOperationRequest
from app.core.exceptions import create_file_exception
from app.services.document_service import get_document_service, DocumentService

router = APIRouter(prefix="/documents", tags=["文档管理"])
logger = logging.getLogger("rag-anything")


@router.delete("/delete", response_model=SuccessResponse, summary="删除文件")
async def delete_files(
    request: FileDeleteRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    删除文件接口
    
    **功能说明:**
    - 支持批量删除文件
    - 可选择是否删除解析数据和向量数据
    - 确保数据完整性，避免孤立数据
    
    **请求参数:**
    - file_ids: 要删除的文件ID列表（必填）
    - delete_parsed_data: 是否删除解析后的数据（默认true）
    - delete_vector_data: 是否删除向量数据（默认true）
    
    **响应数据:**
    - total_files: 总文件数
    - successful_deletions: 成功删除数
    - failed_deletions: 失败删除数
    - results: 每个文件的删除结果
    """
    
    results = []
    successful_deletions = 0
    failed_deletions = 0
    
    for file_id in request.file_ids:
        try:
            # 删除文件
            success = await document_service.delete_file(
                file_id=file_id,
                delete_parsed_data=request.delete_parsed_data,
                delete_vector_data=request.delete_vector_data
            )
            
            if success:
                results.append({
                    "file_id": file_id,
                    "success": True,
                    "message": "文件删除成功"
                })
                successful_deletions += 1
            else:
                results.append({
                    "file_id": file_id,
                    "success": False,
                    "error": "文件删除失败"
                })
                failed_deletions += 1
                
        except Exception as e:
            logger.error(f"删除文件失败: {file_id} - {e}")
            results.append({
                "file_id": file_id,
                "success": False,
                "error": str(e)
            })
            failed_deletions += 1
    
    result_data = {
        "total_files": len(request.file_ids),
        "successful_deletions": successful_deletions,
        "failed_deletions": failed_deletions,
        "delete_parsed_data": request.delete_parsed_data,
        "delete_vector_data": request.delete_vector_data,
        "results": results
    }
    
    logger.info(f"批量删除完成: 成功{successful_deletions}个，失败{failed_deletions}个")
    
    return SuccessResponse(
        data=result_data,
        message=f"批量删除完成，成功删除{successful_deletions}个文件"
    )


@router.post("/parse", response_model=SuccessResponse, summary="解析文档")
async def parse_document(
    request: DocumentProcessRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    解析文档接口
    
    **功能说明:**
    - 使用MinerU解析文档内容
    - 支持多种解析方法：auto、ocr、txt
    - 支持图像、表格、公式处理
    - 异步处理，返回任务ID
    
    **请求参数:**
    - file_id: 文件ID（必填）
    - parse_method: 解析方法（auto/ocr/txt，默认auto）
    - enable_image_processing: 启用图像处理（默认true）
    - enable_table_processing: 启用表格处理（默认true）
    - enable_equation_processing: 启用公式处理（默认true）
    
    **响应数据:**
    - task_id: 解析任务ID
    - file_id: 文件ID
    - parse_method: 使用的解析方法
    - estimated_time: 预估处理时间（秒）
    """
    
    try:
        # 检查文件是否存在
        file_info = await document_service.get_file_info(request.file_id)
        
        # 启动解析任务
        task_id = await document_service.start_parse_task(request.file_id)
        
        result_data = {
            "task_id": task_id,
            "file_id": request.file_id,
            "file_name": file_info.get("original_name"),
            "parse_method": request.parse_method,
            "enable_image_processing": request.enable_image_processing,
            "enable_table_processing": request.enable_table_processing,
            "enable_equation_processing": request.enable_equation_processing,
            "estimated_time": 30  # 预估30秒，实际时间根据文件大小而定
        }
        
        logger.info(f"文档解析任务已启动: {request.file_id} - 任务ID: {task_id}")
        
        return SuccessResponse(
            data=result_data,
            message="文档解析任务已启动"
        )
        
    except Exception as e:
        logger.error(f"启动文档解析任务失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.FILE_PARSE_FAILED,
                f"启动文档解析任务失败: {str(e)}"
            )


@router.post("/index", response_model=SuccessResponse, summary="索引文档到向量数据库")
async def index_document(
    request: VectorStoreRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    索引文档到向量数据库接口
    
    **功能说明:**
    - 将解析后的文档内容向量化
    - 存储到Qdrant向量数据库
    - 支持自定义集合名称
    - 异步处理，返回任务ID
    
    **请求参数:**
    - file_id: 文件ID（必填）
    - collection_name: 集合名称（可选，默认使用文件ID）
    - overwrite: 是否覆盖已有数据（默认false）
    
    **响应数据:**
    - task_id: 索引任务ID
    - file_id: 文件ID
    - collection_name: 使用的集合名称
    - estimated_time: 预估处理时间（秒）
    """
    
    try:
        # 检查文件是否存在
        file_info = await document_service.get_file_info(request.file_id)
        
        # 检查文件是否已解析
        if file_info.get("status") != "parsed":
            raise create_file_exception(
                ErrorCode.INVALID_REQUEST,
                f"文件尚未解析完成，当前状态: {file_info.get('status')}"
            )
        
        # 启动向量化任务（索引到向量数据库）
        task_id = await document_service.start_vectorize_task(request.file_id)
        
        collection_name = request.collection_name or request.file_id
        
        result_data = {
            "task_id": task_id,
            "file_id": request.file_id,
            "file_name": file_info.get("original_name"),
            "collection_name": collection_name,
            "overwrite": request.overwrite,
            "estimated_time": 60  # 预估60秒，实际时间根据文档大小而定
        }
        
        logger.info(f"文档索引任务已启动: {request.file_id} - 任务ID: {task_id}")
        
        return SuccessResponse(
            data=result_data,
            message="文档索引任务已启动"
        )
        
    except Exception as e:
        logger.error(f"启动文档索引任务失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.FILE_PARSE_FAILED,
                f"启动文档索引任务失败: {str(e)}"
            )


@router.get("/{file_id}", response_model=SuccessResponse, summary="获取文件信息")
async def get_file_info(
    file_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文件信息接口
    
    **功能说明:**
    - 获取指定文件的详细信息
    - 包括上传时间、解析状态、索引状态等
    - 显示文件处理的完整生命周期
    
    **路径参数:**
    - file_id: 文件ID（必填）
    
    **响应数据:**
    - file_id: 文件ID
    - original_name: 原始文件名
    - file_size: 文件大小
    - status: 当前状态
    - uploaded_at: 上传时间
    - parsed_at: 解析时间
    - indexed_at: 索引时间
    - metadata: 其他元数据
    """
    
    try:
        file_info = await document_service.get_file_info(file_id)
        
        return SuccessResponse(
            data=file_info,
            message="获取文件信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取文件信息失败: {file_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"获取文件信息失败: {str(e)}"
            )


@router.get("/", response_model=SuccessResponse, summary="列出文件")
async def list_files(
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    列出文件接口
    
    **功能说明:**
    - 获取系统中的文件列表
    - 支持按状态过滤
    - 支持分页查询
    - 按上传时间倒序排列
    
    **查询参数:**
    - status: 文件状态过滤（可选）
    - limit: 每页数量（默认10，最大100）
    - offset: 偏移量（默认0）
    
    **响应数据:**
    - files: 文件列表
    - total_count: 总文件数（当前过滤条件下）
    - has_more: 是否还有更多文件
    """
    
    if limit <= 0 or limit > 100:
        raise create_file_exception(
            ErrorCode.INVALID_REQUEST,
            "每页数量必须在1-100之间"
        )
    
    try:
        files = await document_service.list_files(
            status_filter=status,
            limit=limit,
            offset=offset
        )
        
        # 检查是否还有更多文件
        # 这里可以优化为更精确的计算
        has_more = len(files) == limit
        
        result_data = {
            "files": files,
            "total_count": len(files),  # 这里简化了，实际应该返回总数
            "has_more": has_more,
            "limit": limit,
            "offset": offset,
            "status_filter": status
        }
        
        return SuccessResponse(
            data=result_data,
            message=f"获取文件列表成功，共{len(files)}个文件"
        )
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取文件列表失败: {str(e)}"
        )


@router.get("/categories", response_model=SuccessResponse, summary="获取文件分类统计")
async def get_file_categories(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文件分类统计 - 参考mineru-web的文件管理界面
    
    **功能说明:**
    - 统计各类文件的数量
    - 提供文件分类概览
    - 支持文件类型过滤
    
    **响应数据:**
    - documents: 文档文件数量（PDF、Word、Excel等）
    - images: 图片文件数量
    - texts: 文本文件数量
    - parsed: 已解析文件数量
    - others: 其他文件数量
    """
    try:
        await document_service._get_services()
        categories = await document_service.minio_service.get_file_categories()
        
        return SuccessResponse(
            data=categories,
            message="文件分类统计获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取文件分类失败: {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取文件分类失败: {str(e)}"
        )


@router.get("/{file_id}/preview-url", response_model=SuccessResponse, summary="获取文件预览链接")
async def get_file_preview_url(
    file_id: str,
    expires: int = Query(3600, description="链接有效期（秒）", ge=60, le=86400),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文件预览链接 - 类似mineru-web的文件预览功能
    
    **功能说明:**
    - 生成临时访问链接
    - 支持文件预览和下载
    - 可设置链接有效期
    
    **路径参数:**
    - file_id: 文件ID
    
    **查询参数:**
    - expires: 链接有效期（秒，默认3600）
    
    **响应数据:**
    - preview_url: 预览链接
    - expires_at: 链接过期时间
    - file_info: 文件基本信息
    """
    try:
        # 获取文件信息
        file_info = await document_service.get_file_info(file_id)
        if not file_info:
            raise create_file_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件不存在: {file_id}"
            )
        
        # 生成预览链接
        await document_service._get_services()
        object_name = file_info.get("object_name")
        preview_url = await document_service.minio_service.get_file_url(object_name, expires)
        
        result_data = {
            "file_id": file_id,
            "preview_url": preview_url,
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires)).isoformat(),
            "file_info": {
                "filename": file_info.get("filename"),
                "file_size": file_info.get("file_size"),
                "content_type": file_info.get("content_type"),
                "upload_date": file_info.get("upload_date")
            }
        }
        
        return SuccessResponse(
            data=result_data,
            message="预览链接生成成功"
        )
        
    except Exception as e:
        logger.error(f"获取文件预览链接失败: {file_id} - {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取预览链接失败: {str(e)}"
            )


@router.post("/batch-operations", response_model=SuccessResponse, summary="批量文件操作")
async def batch_file_operations(
    request: BatchFileOperationRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    批量文件操作 - 参考mineru-web的批量处理功能
    
    **功能说明:**
    - 支持批量删除文件
    - 支持批量解析文档
    - 支持批量向量化
    - 返回每个操作的详细结果
    
    **请求体:**
    - operation: 操作类型（delete/parse/vectorize）
    - file_ids: 文件ID列表
    - options: 操作选项
    
    **响应数据:**
    - total_files: 总文件数
    - successful_operations: 成功操作数
    - failed_operations: 失败操作数
    - results: 每个文件的操作结果
    """
    try:
        operation = request.operation
        file_ids = request.file_ids
        options = request.options or {}
        
        if not file_ids:
            raise create_file_exception(
                ErrorCode.INVALID_REQUEST,
                "文件ID列表不能为空"
            )
        
        if len(file_ids) > 50:  # 限制批量操作数量
            raise create_file_exception(
                ErrorCode.INVALID_REQUEST,
                "批量操作最多支持50个文件"
            )
        
        results = []
        successful_operations = 0
        failed_operations = 0
        
        for file_id in file_ids:
            try:
                if operation == "delete":
                    success = await document_service.delete_file(file_id)
                    results.append({
                        "file_id": file_id,
                        "operation": "delete",
                        "success": success,
                        "message": "删除成功" if success else "删除失败"
                    })
                    
                elif operation == "parse":
                    task_id = await document_service.start_parse_task(file_id)
                    results.append({
                        "file_id": file_id,
                        "operation": "parse",
                        "success": True,
                        "task_id": task_id,
                        "message": "解析任务已启动"
                    })
                    
                elif operation == "vectorize":
                    task_id = await document_service.start_vectorize_task(file_id)
                    results.append({
                        "file_id": file_id,
                        "operation": "vectorize",
                        "success": True,
                        "task_id": task_id,
                        "message": "向量化任务已启动"
                    })
                    
                else:
                    results.append({
                        "file_id": file_id,
                        "operation": operation,
                        "success": False,
                        "error": f"不支持的操作类型: {operation}"
                    })
                    failed_operations += 1
                    continue
                
                successful_operations += 1
                
            except Exception as e:
                logger.error(f"批量操作失败: {file_id} - {operation} - {e}")
                results.append({
                    "file_id": file_id,
                    "operation": operation,
                    "success": False,
                    "error": str(e)
                })
                failed_operations += 1
        
        result_data = {
            "operation": operation,
            "total_files": len(file_ids),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "results": results,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"批量操作完成: {operation} - 成功{successful_operations}个，失败{failed_operations}个")
        
        return SuccessResponse(
            data=result_data,
            message=f"批量{operation}操作完成"
        )
        
    except Exception as e:
        logger.error(f"批量文件操作失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"批量操作失败: {str(e)}"
            )


@router.get("/dashboard/stats", response_model=SuccessResponse, summary="获取文件管理仪表板统计")
async def get_dashboard_stats(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文件管理仪表板统计 - 类似mineru-web的首页概览
    
    **功能说明:**
    - 文件总数和分类统计
    - 解析任务统计
    - 存储使用情况
    - 最近活动记录
    
    **响应数据:**
    - file_stats: 文件统计信息
    - task_stats: 任务统计信息
    - storage_stats: 存储统计信息
    - recent_activities: 最近活动
    """
    try:
        await document_service._get_services()
        
        # 获取文件分类统计
        file_categories = await document_service.minio_service.get_file_categories()
        
        # 获取任务统计
        task_stats = await document_service.cache_service.get_queue_stats("document_parse")
        
        # 获取文件列表用于计算存储统计
        files = await document_service.list_files(limit=1000)  # 获取更多文件用于统计
        
        total_size = sum(file.get("file_size", 0) for file in files)
        
        # 计算解析成功率
        parsed_files = sum(1 for file in files if file.get("parse_status") == "completed")
        parse_success_rate = (parsed_files / len(files) * 100) if files else 0
        
        # 最近上传的文件（最近7天）
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_files = [
            file for file in files 
            if datetime.fromisoformat(file.get("upload_date", "1970-01-01")) > recent_cutoff
        ]
        
        dashboard_data = {
            "file_stats": {
                "total_files": len(files),
                "categories": file_categories,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "parse_success_rate": round(parse_success_rate, 1),
                "recent_uploads": len(recent_files)
            },
            "task_stats": task_stats,
            "storage_stats": {
                "used_space_mb": round(total_size / (1024 * 1024), 2),
                "file_count": len(files),
                "avg_file_size_mb": round(total_size / len(files) / (1024 * 1024), 2) if files else 0
            },
            "recent_activities": [
                {
                    "type": "upload",
                    "filename": file.get("filename"),
                    "upload_date": file.get("upload_date"),
                    "file_size": file.get("file_size")
                }
                for file in sorted(recent_files, key=lambda x: x.get("upload_date", ""), reverse=True)[:10]
            ]
        }
        
        return SuccessResponse(
            data=dashboard_data,
            message="仪表板统计获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取仪表板统计失败: {e}")
        raise create_file_exception(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取仪表板统计失败: {str(e)}"
        ) 