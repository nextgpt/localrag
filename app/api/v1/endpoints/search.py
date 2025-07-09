"""
检索和搜索API端点
处理向量检索、语义检索和问答生成
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
import logging

from app.models.responses import SuccessResponse, PaginatedResponse, PaginationInfo, ErrorCode
from app.models.requests import SearchRequest, SearchType
from app.core.exceptions import create_search_exception
from app.services.search_service import get_search_service, SearchService

router = APIRouter(prefix="/search", tags=["检索搜索"])
logger = logging.getLogger("rag-anything")


@router.post("/", response_model=PaginatedResponse, summary="统一检索接口")
async def search_documents(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    统一检索接口
    
    **功能说明:**
    - 支持三种检索类型：向量检索、语义检索、混合检索
    - 支持分页查询
    - 支持按文件ID过滤
    - 返回相关性分数和元数据
    
    **请求参数:**
    - query: 检索查询（必填）
    - search_type: 检索类型（vector/semantic/hybrid，默认hybrid）
    - limit: 返回结果数量（1-100，默认10）
    - offset: 结果偏移量（默认0）
    - file_ids: 限制检索的文件ID列表（可选）
    
    **响应数据:**
    - results: 检索结果列表
    - pagination: 分页信息
    - search_metadata: 检索元数据
    """
    
    try:
        # 执行检索，获取更多结果用于分页
        extended_limit = request.limit + request.offset
        results = await search_service.search(
            query=request.query,
            search_type=request.search_type,
            limit=extended_limit,
            score_threshold=0.5,  # 🔧 降低阈值以获得更多相关结果
            file_ids=request.file_ids
        )
        
        # 应用分页逻辑
        total_count = len(results)
        start_idx = request.offset
        end_idx = request.offset + request.limit
        result_data = results[start_idx:end_idx]
        
        # 计算分页信息
        pages = (total_count + request.limit - 1) // request.limit
        pagination = PaginationInfo(
            page=(request.offset // request.limit) + 1,
            size=request.limit,
            total=total_count,
            pages=pages
        )
        
        # 检索元数据
        search_metadata = {
            "search_type": request.search_type,
            "query_length": len(request.query),
            "result_count": len(results),
            "file_filter_count": len(request.file_ids) if request.file_ids else 0
        }
        
        logger.info(f"检索完成: 查询='{request.query}', 类型={request.search_type}, 结果数={len(results)}")
        
        return PaginatedResponse(
            data={
                "results": result_data,
                "search_metadata": search_metadata
            },
            pagination=pagination,
            message=f"检索完成，找到{total_count}个相关结果"
        )
        
    except Exception as e:
        logger.error(f"检索失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_search_exception(
                ErrorCode.SEARCH_FAILED,
                f"检索失败: {str(e)}"
            )


@router.post("/vector", response_model=SuccessResponse, summary="向量检索")
async def vector_search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    向量检索接口
    
    **功能说明:**
    - 基于语义相似度的向量检索
    - 使用嵌入模型计算查询向量
    - 在向量数据库中查找最相似的内容
    - 返回相似度分数和距离信息
    
    **适用场景:**
    - 语义相似内容查找
    - 概念匹配
    - 主题相关性检索
    """
    
    try:
        # 强制使用向量检索
        request.search_type = SearchType.VECTOR
        
        results = await search_service.vector_search(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            file_ids=request.file_ids
        )
        
        # 结果已经是Dict格式
        result_data = results
        
        return SuccessResponse(
            data={
                "results": result_data,
                "search_type": "vector",
                "total_results": len(results)
            },
            message=f"向量检索完成，找到{len(results)}个结果"
        )
        
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_search_exception(
                ErrorCode.SEARCH_FAILED,
                f"向量检索失败: {str(e)}"
            )


@router.post("/semantic", response_model=SuccessResponse, summary="语义检索")
async def semantic_search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    语义检索接口
    
    **功能说明:**
    - 基于知识图谱的语义检索
    - 理解实体关系和语义结构
    - 进行推理和知识关联
    - 返回语义相关性分数
    
    **适用场景:**
    - 复杂问题回答
    - 知识推理
    - 关系查询
    """
    
    try:
        # 强制使用语义检索
        request.search_type = SearchType.SEMANTIC
        
        results = await search_service.semantic_search(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            file_ids=request.file_ids
        )
        
        # 结果已经是Dict格式
        result_data = results
        
        return SuccessResponse(
            data={
                "results": result_data,
                "search_type": "semantic",
                "total_results": len(results)
            },
            message=f"语义检索完成，找到{len(results)}个结果"
        )
        
    except Exception as e:
        logger.error(f"语义检索失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_search_exception(
                ErrorCode.SEARCH_FAILED,
                f"语义检索失败: {str(e)}"
            )


@router.post("/hybrid", response_model=SuccessResponse, summary="混合检索")
async def hybrid_search(
    request: SearchRequest,
    vector_weight: float = 0.6,
    semantic_weight: float = 0.4,
    search_service: SearchService = Depends(get_search_service)
):
    """
    混合检索接口
    
    **功能说明:**
    - 结合向量检索和语义检索的优势
    - 可调整两种检索方法的权重
    - 智能合并和排序结果
    - 提供最佳的检索效果
    
    **查询参数:**
    - vector_weight: 向量检索权重（0-1，默认0.6）
    - semantic_weight: 语义检索权重（0-1，默认0.4）
    
    **适用场景:**
    - 综合性查询
    - 平衡精确性和召回率
    - 复杂信息检索
    """
    
    # 验证权重参数
    if vector_weight + semantic_weight != 1.0:
        raise create_search_exception(
            ErrorCode.INVALID_SEARCH_PARAMS,
            f"权重之和必须为1.0，当前为{vector_weight + semantic_weight}"
        )
    
    try:
        # 强制使用混合检索
        request.search_type = SearchType.HYBRID
        
        results = await search_service.hybrid_search(
            query=request.query,
            limit=request.limit,
            file_ids=request.file_ids,
            vector_weight=vector_weight,
            text_weight=semantic_weight  # 修复参数名映射
        )
        
        # 结果已经是Dict格式
        result_data = results
        
        return SuccessResponse(
            data={
                "results": result_data,
                "search_type": "hybrid",
                "vector_weight": vector_weight,
                "semantic_weight": semantic_weight,
                "total_results": len(results)
            },
            message=f"混合检索完成，找到{len(results)}个结果"
        )
        
    except Exception as e:
        logger.error(f"混合检索失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_search_exception(
                ErrorCode.SEARCH_FAILED,
                f"混合检索失败: {str(e)}"
            )


@router.post("/answer", response_model=SuccessResponse, summary="问答生成")
async def generate_answer(
    request: SearchRequest,
    include_sources: bool = True,
    search_service: SearchService = Depends(get_search_service)
):
    """
    问答生成接口
    
    **功能说明:**
    - 基于检索结果生成自然语言答案
    - 结合多模态内容（文本、图像、表格）
    - 支持上下文引用和来源标注
    - 使用大语言模型进行推理和生成
    
    **查询参数:**
    - include_sources: 是否包含来源信息（默认true）
    
    **适用场景:**
    - 问答系统
    - 智能客服
    - 知识查询
    """
    
    try:
        # 首先执行检索
        results = await search_service.search(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit,
            score_threshold=0.5,  # 🔧 降低阈值以获得更多相关结果
            file_ids=request.file_ids
        )
        
        if not results:
            return SuccessResponse(
                data={
                    "query": request.query,
                    "answer": "抱歉，没有找到相关信息来回答您的问题。",
                    "context_count": 0,
                    "sources": [] if include_sources else None
                },
                message="未找到相关信息"
            )
        
        # 生成答案
        answer_data = await search_service.generate_answer(
            query=request.query,
            search_results=results,
            include_sources=include_sources
        )
        
        logger.info(f"问答生成完成: 查询='{request.query}', 上下文数={answer_data['context_count']}")
        
        return SuccessResponse(
            data=answer_data,
            message="答案生成完成"
        )
        
    except Exception as e:
        logger.error(f"问答生成失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_search_exception(
                ErrorCode.SEARCH_FAILED,
                f"问答生成失败: {str(e)}"
            )


@router.get("/stats", response_model=SuccessResponse, summary="检索统计信息")
async def get_search_stats(
    search_service: SearchService = Depends(get_search_service)
):
    """
    检索统计信息接口
    
    **功能说明:**
    - 获取系统的检索统计信息
    - 包括文档数量、索引状态等
    - 提供系统性能和容量信息
    
    **响应数据:**
    - total_documents: 总文档数
    - indexed_documents: 已索引文档数
    - total_chunks: 总内容块数
    - vector_dimensions: 向量维度
    - search_types: 支持的检索类型
    """
    
    try:
        stats = await search_service.get_search_statistics()
        
        return SuccessResponse(
            data=stats,
            message="获取检索统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取检索统计信息失败: {e}")
        raise create_search_exception(
            ErrorCode.SEARCH_FAILED,
            f"获取检索统计信息失败: {str(e)}"
        ) 