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
    ResponseModel, ErrorCode, create_error_response, create_success_response
)
from app.services.knowledge_base_service import get_knowledge_base_service
from app.services.document_service import get_document_service
from app.services.search_service import SearchService
from app.core.config import settings

logger = logging.getLogger("rag-anything")
router = APIRouter()


@router.post("/", response_model=ResponseModel[KnowledgeBase])
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
        
        return create_success_response(
            data=knowledge_base,
            message=f"知识库创建成功: {knowledge_base.name}"
        )
        
    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"创建知识库失败: {str(e)}"
        )


@router.get("/", response_model=ResponseModel[List[KnowledgeBase]])
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
        
        return create_success_response(
            data=knowledge_bases,
            message=f"获取到 {len(knowledge_bases)} 个知识库，共 {total} 个"
        )
        
    except Exception as e:
        logger.error(f"列出知识库失败: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"列出知识库失败: {str(e)}"
        )


@router.get("/{kb_id}", response_model=ResponseModel[KnowledgeBase])
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
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        return create_success_response(
            data=knowledge_base,
            message="获取知识库信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取知识库失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取知识库失败: {str(e)}"
        )


@router.put("/{kb_id}", response_model=ResponseModel[KnowledgeBase])
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
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        return create_success_response(
            data=knowledge_base,
            message="知识库更新成功"
        )
        
    except Exception as e:
        logger.error(f"更新知识库失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"更新知识库失败: {str(e)}"
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
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        return create_success_response(
            data={"deleted": True, "files_deleted": delete_files},
            message="知识库删除成功"
        )
        
    except Exception as e:
        logger.error(f"删除知识库失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"删除知识库失败: {str(e)}"
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
            return create_error_response(
                ErrorCode.INVALID_REQUEST,
                "添加文件到知识库失败，请检查知识库和文件是否存在"
            )
        
        return create_success_response(
            data={"kb_id": kb_id, "file_id": file_id},
            message="文件添加到知识库成功"
        )
        
    except Exception as e:
        logger.error(f"添加文件到知识库失败: {kb_id}/{file_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"添加文件失败: {str(e)}"
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
            return create_error_response(
                ErrorCode.NOT_FOUND,
                "移除文件失败，请检查知识库和文件关联"
            )
        
        return create_success_response(
            data={"kb_id": kb_id, "file_id": file_id},
            message="文件从知识库移除成功"
        )
        
    except Exception as e:
        logger.error(f"移除文件失败: {kb_id}/{file_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"移除文件失败: {str(e)}"
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
        
        return create_success_response(
            data=files,
            message=f"知识库包含 {len(files)} 个文件"
        )
        
    except Exception as e:
        logger.error(f"获取知识库文件失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取知识库文件失败: {str(e)}"
        )


@router.get("/{kb_id}/stats", response_model=ResponseModel[KnowledgeBaseStats])
async def get_knowledge_base_stats(
    kb_id: str = Path(..., description="知识库ID")
):
    """
    获取知识库统计信息
    
    包括文件数量、向量数量、处理状态分布等
    """
    try:
        service = await get_knowledge_base_service()
        stats = await service.get_knowledge_base_stats(kb_id)
        
        if not stats:
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        return create_success_response(
            data=stats,
            message="获取知识库统计成功"
        )
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"获取知识库统计失败: {str(e)}"
        )


@router.post("/{kb_id}/search")
async def search_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    request: KnowledgeBaseSearch = Body(...)
):
    """
    在知识库中进行语义检索
    
    支持文本和图片结果，可配置检索参数
    """
    try:
        # 验证知识库存在
        kb_service = await get_knowledge_base_service()
        knowledge_base = await kb_service.get_knowledge_base(kb_id)
        
        if not knowledge_base:
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        # 使用知识库配置的检索参数（可被请求参数覆盖）
        top_k = request.top_k or knowledge_base.qdrant_config.top_k
        score_threshold = request.score_threshold or knowledge_base.qdrant_config.score_threshold
        
        # 执行检索
        search_service = SearchService()
        
        # 基于知识库的专用检索
        results = await search_service.search_in_knowledge_base(
            kb_id=kb_id,
            collection_name=knowledge_base.qdrant_config.collection_name,
            query=request.query,
            top_k=top_k,
            score_threshold=score_threshold,
            return_images=request.return_images,
            return_metadata=request.return_metadata,
            file_types=request.file_types,
            date_range=request.date_range
        )
        
        return create_success_response(
            data={
                "kb_id": kb_id,
                "query": request.query,
                "results": results,
                "search_params": {
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "return_images": request.return_images
                }
            },
            message=f"检索完成，找到 {len(results)} 个相关结果"
        )
        
    except Exception as e:
        logger.error(f"知识库检索失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"知识库检索失败: {str(e)}"
        )


@router.post("/{kb_id}/vectorize")
async def vectorize_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    priority: int = Query(0, description="处理优先级")
):
    """
    对知识库中的所有文件进行向量化
    
    批量处理知识库中已解析的文件
    """
    try:
        kb_service = await get_knowledge_base_service()
        doc_service = await get_document_service()
        
        # 获取知识库文件列表
        file_ids = await kb_service.get_knowledge_base_files(kb_id)
        
        if not file_ids:
            return create_error_response(
                ErrorCode.INVALID_REQUEST,
                "知识库中没有文件需要向量化"
            )
        
        # 启动向量化任务
        task_ids = []
        for file_id in file_ids:
            try:
                task_id = await doc_service.start_vectorize_task(file_id, priority)
                task_ids.append(task_id)
            except Exception as e:
                logger.warning(f"文件 {file_id} 向量化任务启动失败: {e}")
        
        return create_success_response(
            data={
                "kb_id": kb_id,
                "task_ids": task_ids,
                "total_files": len(file_ids),
                "started_tasks": len(task_ids)
            },
            message=f"启动向量化任务：{len(task_ids)}/{len(file_ids)} 个文件"
        )
        
    except Exception as e:
        logger.error(f"知识库向量化失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"知识库向量化失败: {str(e)}"
        )


@router.post("/{kb_id}/reindex")
async def reindex_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    optimize_config: Optional[Dict[str, Any]] = Body(None, description="优化配置")
):
    """
    重建知识库索引
    
    可配置HNSW等专业参数优化检索性能
    """
    try:
        kb_service = await get_knowledge_base_service()
        knowledge_base = await kb_service.get_knowledge_base(kb_id)
        
        if not knowledge_base:
            return create_error_response(
                ErrorCode.NOT_FOUND,
                f"知识库不存在: {kb_id}"
            )
        
        # TODO: 实现索引重建逻辑
        # 1. 备份现有向量
        # 2. 创建新集合（应用新配置）
        # 3. 迁移向量数据
        # 4. 切换集合
        
        return create_success_response(
            data={"kb_id": kb_id, "status": "reindexing"},
            message="索引重建任务已启动（功能开发中）"
        )
        
    except Exception as e:
        logger.error(f"重建索引失败: {kb_id} - {e}")
        return create_error_response(
            ErrorCode.INTERNAL_SERVER_ERROR,
            f"重建索引失败: {str(e)}"
        ) 