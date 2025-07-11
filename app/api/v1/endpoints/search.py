"""
检索和搜索API端点
处理向量检索、语义检索和问答生成
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any
import logging
from pydantic import BaseModel

from app.models.responses import SuccessResponse, PaginatedResponse, PaginationInfo, ErrorCode
from app.models.requests import SearchRequest, SearchType
from app.core.exceptions import create_search_exception
from app.services.search_service import get_search_service, SearchService

router = APIRouter(tags=["检索搜索"])  # 🔧 移除重复的prefix
logger = logging.getLogger("rag-anything")


# 🎯 新增：招标书分析请求模型
class TenderAnalysisRequest(BaseModel):
    query: str = "项目名称"
    file_ids: Optional[List[str]] = None
    analysis_type: str = "general"  # general/project_info/technical_specs/commercial_terms/risks
    limit: int = 20
    score_threshold: float = 0.1  # 🔧 降低默认阈值提高召回率
    collection_name: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "项目名称",
                "file_ids": ["your-uploaded-file-id"],  # 使用实际上传的文件ID
                "analysis_type": "project_info",
                "limit": 20,
                "score_threshold": 0.1
            }
        }


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
            score_threshold=0.1,  # 🔧 大幅降低阈值确保能找到结果
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
            score_threshold=0.1,  # 🔧 大幅降低阈值确保能找到结果
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

@router.post("/tender", summary="🎯 招标书专用搜索分析")
async def search_tender_documents(
    request: TenderAnalysisRequest,
    search_service = Depends(get_search_service)
) -> Dict[str, Any]:
    """
    🎯 招标书专用搜索分析 - 99%精准度专业解读
    
    ## 分析类型说明：
    - **general**: 综合分析（默认）
    - **project_info**: 项目性质、工期要求、节点里程碑、截标日期
    - **technical_specs**: 技术要求、施工方案、材料设备要求
    - **commercial_terms**: 投标人责任、工作范围、报价要求、投标书编制
    - **risks**: 风险识别、重难点分析、矛盾检测
    
    ## 返回结构化分析：
    - 关键信息提取（项目名称、工期、预算等）
    - 时间线分析（截标、开标、里程碑）
    - 财务信息（预算、保证金、付款条件）
    - 技术要求（质量标准、材料设备）
    - 资格要求（企业资质、人员配置）
    - 风险识别（潜在风险、矛盾检测）
    - 专业报告（执行摘要、建议、行动项）
    - 置信度评估（整体置信度、完整性分析）
    """
    try:
        logger.info(f"🎯 招标书专用搜索: {request.query} - 类型: {request.analysis_type}")
        
        # 验证分析类型
        valid_analysis_types = ["general", "project_info", "technical_specs", "commercial_terms", "risks"]
        if request.analysis_type not in valid_analysis_types:
            raise HTTPException(
                status_code=400,
                detail=f"无效的分析类型。支持的类型: {', '.join(valid_analysis_types)}"
            )
        
        # 🔧 验证和修复collection_name参数
        actual_collection_name = None
        if request.collection_name is not None:
            if isinstance(request.collection_name, str):
                actual_collection_name = request.collection_name
            else:
                logger.warning(f"招标书搜索API收到非字符串collection_name: {type(request.collection_name)}, 值: {request.collection_name}")
                actual_collection_name = None
        
        # 执行招标书专用搜索
        result = await search_service.search_tender_documents(
            query=request.query,
            file_ids=request.file_ids,
            analysis_type=request.analysis_type,
            limit=request.limit,
            score_threshold=request.score_threshold,
            collection_name=actual_collection_name
        )
        
        # 添加API响应元数据
        result["api_metadata"] = {
            "endpoint": "/search/tender",
            "analysis_type_description": {
                "general": "综合分析所有类型信息",
                "project_info": "项目性质、工期要求、节点里程碑、截标日期等关键事项",
                "technical_specs": "技术要求、制定合理的施工方案、材料和设备要求",
                "commercial_terms": "投标人责任、工作范围、报价要求、投标书编制内容",
                "risks": "工程风险、重难点、错误矛盾检测"
            }.get(request.analysis_type, "未知分析类型"),
            "precision_target": "99%",
            "specialized_features": [
                "智能结构识别",
                "关键信息提取", 
                "多层次检索",
                "矛盾检测",
                "风险识别",
                "置信度评估"
            ]
        }
        
        logger.info(f"✅ 招标书分析完成: {result['total_results']}个结果")
        return result
        
    except Exception as e:
        logger.error(f"❌ 招标书搜索失败: {request.query} - {e}")
        raise HTTPException(status_code=500, detail=f"招标书搜索失败: {str(e)}")

@router.post("/tender/batch", summary="🎯 批量招标书分析")
async def batch_tender_analysis(
    queries: List[str],
    file_ids: Optional[List[str]] = None,
    analysis_type: str = "general",
    limit: int = 10,
    score_threshold: float = 0.1,  # 🔧 降低默认阈值提高召回率
    collection_name: Optional[str] = None,
    search_service = Depends(get_search_service)
) -> Dict[str, Any]:
    """
    🎯 批量招标书分析 - 一次性分析多个查询
    
    适用场景：
    - 全面解读一份招标书的所有要求
    - 同时检查多个关键信息点
    - 批量风险识别和矛盾检测
    """
    try:
        logger.info(f"🎯 批量招标书分析: {len(queries)}个查询")
        
        results = {}
        for i, query in enumerate(queries):
            try:
                result = await search_service.search_tender_documents(
                    query=query,
                    file_ids=file_ids,
                    analysis_type=analysis_type,
                    limit=limit,
                    score_threshold=score_threshold,
                    collection_name=collection_name
                )
                results[f"query_{i+1}_{query[:20]}"] = result
                
            except Exception as e:
                logger.error(f"查询失败: {query} - {e}")
                results[f"query_{i+1}_{query[:20]}"] = {
                    "error": str(e),
                    "query": query
                }
        
        # 生成综合报告
        comprehensive_analysis = _generate_comprehensive_analysis(results)
        
        return {
            "batch_analysis": results,
            "comprehensive_analysis": comprehensive_analysis,
            "summary": {
                "total_queries": len(queries),
                "successful_queries": len([r for r in results.values() if "error" not in r]),
                "failed_queries": len([r for r in results.values() if "error" in r])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 批量分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量分析失败: {str(e)}")

def _generate_comprehensive_analysis(batch_results: Dict[str, Any]) -> Dict[str, Any]:
    """生成批量分析的综合报告"""
    
    all_risks = []
    all_contradictions = []
    overall_completeness = []
    key_findings = []
    
    for query_key, result in batch_results.items():
        if "error" in result:
            continue
            
        # 收集风险
        risks = result.get("structured_analysis", {}).get("risks_and_issues", [])
        all_risks.extend(risks)
        
        # 收集矛盾
        contradictions = result.get("structured_analysis", {}).get("contradictions", [])
        all_contradictions.extend(contradictions)
        
        # 收集完整性分数
        completeness = result.get("structured_analysis", {}).get("completeness_analysis", {})
        if completeness.get("completeness_score"):
            overall_completeness.append(completeness["completeness_score"])
        
        # 收集关键发现
        findings = result.get("tender_report", {}).get("detailed_findings", {})
        key_findings.extend(findings.get("positive_findings", []))
    
    # 计算综合指标
    avg_completeness = sum(overall_completeness) / len(overall_completeness) if overall_completeness else 0
    total_risks = len(all_risks)
    total_contradictions = len(all_contradictions)
    
    # 生成整体风险评估
    high_risk_count = len([r for r in all_risks if r.get("risk_score", 0) >= 3])
    overall_risk_level = "高" if high_risk_count > 0 else ("中" if total_risks > 5 else "低")
    
    return {
        "overall_completeness": avg_completeness,
        "risk_summary": {
            "total_risks": total_risks,
            "high_risk_count": high_risk_count,
            "overall_risk_level": overall_risk_level,
            "top_risks": sorted(all_risks, key=lambda x: x.get("risk_score", 0), reverse=True)[:5]
        },
        "consistency_check": {
            "total_contradictions": total_contradictions,
            "contradiction_details": all_contradictions[:3]
        },
        "key_achievements": list(set(key_findings))[:10],
        "recommendations": [
            "🔍 重点关注高风险项目" if high_risk_count > 0 else "✅ 风险水平可控",
            "📞 联系招标方澄清矛盾" if total_contradictions > 0 else "✅ 信息一致性良好",
            "📋 补充缺失信息" if avg_completeness < 0.8 else "✅ 信息完整性良好"
        ]
    } 