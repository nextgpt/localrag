"""
知识库管理API
提供知识库的创建、管理、文件导入、检索等功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from typing import List, Optional, Dict, Any
import logging

from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate, 
    KnowledgeBaseSearch, KnowledgeBaseStatus, KnowledgeBaseStats
)
from app.models.responses import (
    SuccessResponse, ErrorResponse, ErrorCode, ErrorDetail
)
from app.services.knowledge_base_service import get_knowledge_base_service
from app.services.document_service import get_document_service
from app.services.search_service import SearchService
from app.core.config import settings

logger = logging.getLogger("rag-anything")
router = APIRouter()


@router.post("/", response_model=SuccessResponse)
async def create_knowledge_base(
    request: KnowledgeBaseCreate
):
    """
    创建新的知识库
    
    创建一个独立的向量集合，用于存储特定领域的文档
    """
    try:
        service = await get_knowledge_base_service()
        knowledge_base = await service.create_knowledge_base(request)
        
        return SuccessResponse(
            data=knowledge_base,
            message=f"知识库创建成功: {knowledge_base.name}"
        )
        
    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"创建知识库失败: {str(e)}"
            )
        )


@router.get("/", response_model=SuccessResponse)
async def list_knowledge_bases(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    status: Optional[KnowledgeBaseStatus] = Query(None, description="状态过滤")
):
    """
    列出知识库
    
    支持分页和状态过滤
    """
    try:
        service = await get_knowledge_base_service()
        knowledge_bases, total = await service.list_knowledge_bases(
            limit=limit, 
            offset=offset, 
            status_filter=status
        )
        
        return SuccessResponse(
            data=knowledge_bases,
            message=f"获取到 {len(knowledge_bases)} 个知识库，共 {total} 个"
        )
        
    except Exception as e:
        logger.error(f"列出知识库失败: {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"列出知识库失败: {str(e)}"
            )
        )


@router.get("/{kb_id}", response_model=SuccessResponse)
async def get_knowledge_base(
    kb_id: str = Path(..., description="知识库ID")
):
    """
    获取特定知识库的详细信息
    """
    try:
        service = await get_knowledge_base_service()
        knowledge_base = await service.get_knowledge_base(kb_id)
        
        if not knowledge_base:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        return SuccessResponse(
            data=knowledge_base,
            message="获取知识库信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取知识库失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"获取知识库失败: {str(e)}"
            )
        )


@router.put("/{kb_id}", response_model=SuccessResponse)
async def update_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    request: KnowledgeBaseUpdate = Body(...)
):
    """
    更新知识库配置
    
    可以调整检索参数、描述等信息
    """
    try:
        service = await get_knowledge_base_service()
        knowledge_base = await service.update_knowledge_base(kb_id, request)
        
        if not knowledge_base:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        return SuccessResponse(
            data=knowledge_base,
            message="知识库更新成功"
        )
        
    except Exception as e:
        logger.error(f"更新知识库失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"更新知识库失败: {str(e)}"
            )
        )


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    delete_files: bool = Query(False, description="是否同时删除文件")
):
    """
    删除知识库
    
    可选择是否同时删除关联的文件
    """
    try:
        service = await get_knowledge_base_service()
        success = await service.delete_knowledge_base(kb_id, delete_files)
        
        if not success:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        return SuccessResponse(
            data={"deleted": True, "files_deleted": delete_files},
            message="知识库删除成功"
        )
        
    except Exception as e:
        logger.error(f"删除知识库失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"删除知识库失败: {str(e)}"
            )
        )


@router.post("/{kb_id}/files/{file_id}")
async def add_file_to_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    file_id: str = Path(..., description="文件ID")
):
    """
    将文件添加到知识库
    
    文件需要先上传和解析完成
    """
    try:
        service = await get_knowledge_base_service()
        success = await service.add_file_to_knowledge_base(kb_id, file_id)
        
        if not success:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.INVALID_REQUEST,
                    message="添加文件到知识库失败，请检查知识库和文件是否存在"
                )
            )
        
        return SuccessResponse(
            data={"kb_id": kb_id, "file_id": file_id},
            message="文件添加到知识库成功"
        )
        
    except Exception as e:
        logger.error(f"添加文件到知识库失败: {kb_id}/{file_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"添加文件失败: {str(e)}"
            )
        )


@router.delete("/{kb_id}/files/{file_id}")
async def remove_file_from_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    file_id: str = Path(..., description="文件ID")
):
    """
    从知识库中移除文件
    
    只移除关联关系，不删除原文件
    """
    try:
        service = await get_knowledge_base_service()
        success = await service.remove_file_from_knowledge_base(kb_id, file_id)
        
        if not success:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message="移除文件失败，请检查知识库和文件关联"
                )
            )
        
        return SuccessResponse(
            data={"kb_id": kb_id, "file_id": file_id},
            message="文件从知识库移除成功"
        )
        
    except Exception as e:
        logger.error(f"移除文件失败: {kb_id}/{file_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"移除文件失败: {str(e)}"
            )
        )


@router.get("/{kb_id}/files")
async def get_knowledge_base_files(
    kb_id: str = Path(..., description="知识库ID")
):
    """
    获取知识库中的所有文件
    """
    try:
        service = await get_knowledge_base_service()
        file_ids = await service.get_knowledge_base_files(kb_id)
        
        # 获取文件详细信息
        doc_service = await get_document_service()
        files = []
        for file_id in file_ids:
            file_info = await doc_service.get_file_info(file_id)
            if file_info:
                files.append(file_info)
        
        return SuccessResponse(
            data=files,
            message=f"知识库包含 {len(files)} 个文件"
        )
        
    except Exception as e:
        logger.error(f"获取知识库文件失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"获取知识库文件失败: {str(e)}"
            )
        )


@router.get("/{kb_id}/stats", response_model=SuccessResponse)
async def get_knowledge_base_stats(
    kb_id: str = Path(..., description="知识库ID")
):
    """
    获取知识库统计信息
    """
    try:
        service = await get_knowledge_base_service()
        stats = await service.get_knowledge_base_stats(kb_id)
        
        if not stats:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        return SuccessResponse(
            data=stats,
            message="获取知识库统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"获取知识库统计失败: {str(e)}"
            )
        )


@router.post("/{kb_id}/search")
async def search_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    request: KnowledgeBaseSearch = Body(...)
):
    """
    在知识库中搜索
    
    支持返回文本和图片结果
    """
    try:
        service = await get_knowledge_base_service()
        
        # 验证知识库是否存在
        kb = await service.get_knowledge_base(kb_id)
        if not kb:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        # 执行搜索
        search_service = SearchService()
        results = await search_service.search_knowledge_base(
            kb_id=kb_id,
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            include_images=request.include_images
        )
        
        return SuccessResponse(
            data={
                "query": request.query,
                "results": results,
                "total_results": len(results),
                "search_type": request.search_type,
                "knowledge_base": kb_id
            },
            message=f"搜索完成，找到 {len(results)} 个结果"
        )
        
    except Exception as e:
        logger.error(f"知识库搜索失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"搜索失败: {str(e)}"
            )
        )


@router.post("/{kb_id}/vectorize")
async def vectorize_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    priority: int = Query(0, description="处理优先级")
):
    """
    批量向量化知识库中的文件
    
    对知识库中所有已解析的文件进行向量化处理
    """
    try:
        service = await get_knowledge_base_service()
        
        # 验证知识库是否存在
        kb = await service.get_knowledge_base(kb_id)
        if not kb:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        # 启动批量向量化
        task_ids = await service.vectorize_knowledge_base(kb_id, priority)
        
        return SuccessResponse(
            data={
                "knowledge_base_id": kb_id,
                "task_ids": task_ids,
                "total_tasks": len(task_ids),
                "priority": priority
            },
            message=f"已启动 {len(task_ids)} 个向量化任务"
        )
        
    except Exception as e:
        logger.error(f"知识库向量化失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"向量化失败: {str(e)}"
            )
        )


@router.post("/{kb_id}/reindex")
async def reindex_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    optimize_config: Optional[Dict[str, Any]] = Body(None, description="优化配置")
):
    """
    重建知识库索引
    
    删除现有向量数据，重新向量化所有文件
    支持HNSW参数优化
    """
    try:
        service = await get_knowledge_base_service()
        
        # 验证知识库是否存在
        kb = await service.get_knowledge_base(kb_id)
        if not kb:
            return ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.NOT_FOUND,
                    message=f"知识库不存在: {kb_id}"
                )
            )
        
        # 执行重建索引
        success = await service.reindex_knowledge_base(kb_id, optimize_config)
        
        return SuccessResponse(
            data={
                "knowledge_base_id": kb_id,
                "reindex_success": success,
                "optimize_config": optimize_config
            },
            message="知识库索引重建成功"
        )
        
    except Exception as e:
        logger.error(f"知识库索引重建失败: {kb_id} - {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"索引重建失败: {str(e)}"
            )
        ) 