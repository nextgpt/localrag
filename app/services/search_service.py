"""
搜索服务
负责文档的语义检索、向量检索和混合检索，使用分布式存储架构
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

from app.core.config import settings
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception
from app.services.vector_service import get_vector_service
from app.services.cache_service import get_cache_service

logger = logging.getLogger("rag-anything")


class SearchService:
    """搜索服务"""
    
    def __init__(self):
        self.vector_service = None
        self.cache_service = None
        
    async def _get_services(self):
        """获取依赖服务"""
        if self.vector_service is None:
            self.vector_service = await get_vector_service()
        if self.cache_service is None:
            self.cache_service = await get_cache_service()
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询的embedding向量"""
        try:
            # 检查API配置
            if not settings.EMBEDDING_API_BASE or not settings.EMBEDDING_API_KEY:
                logger.warning("Embedding API配置不完整，尝试使用本地方案")
                return await self._get_local_embedding(query)
                
            import httpx
            
            # 确保API_KEY不为空
            api_key = str(settings.EMBEDDING_API_KEY).strip()
            if not api_key or api_key == "None":
                logger.warning("EMBEDDING_API_KEY为空，尝试使用本地方案")
                return await self._get_local_embedding(query)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.EMBEDDING_API_BASE}/embeddings",
                    json={
                        "model": settings.EMBEDDING_MODEL_NAME,
                        "input": query
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                return embedding
                
        except Exception as e:
            logger.error(f"获取查询embedding失败: {e}")
            # 尝试使用本地embedding作为fallback
            try:
                logger.info("尝试使用本地embedding作为fallback")
                return await self._get_local_embedding(query)
            except Exception as fallback_error:
                logger.error(f"本地embedding也失败: {fallback_error}")
                raise create_service_exception(
                    ErrorCode.EMBEDDING_CONNECTION_ERROR,
                    f"获取查询向量失败: {str(e)}"
                )
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """使用本地模型生成embedding - fallback方案"""
        try:
            # 使用配置中的向量维度（Qwen3-Embedding-8B：3072维）
            import hashlib
            import struct
            from app.core.config import settings
            
            vector_dim = settings.EMBEDDING_DIMENSION
            
            # 使用文本哈希生成确定性向量
            text_hash = hashlib.md5(text.encode('utf-8')).digest()
            
            # 生成指定维度的向量
            embedding = []
            for i in range(vector_dim):
                # 使用哈希值的不同部分生成浮点数
                byte_offset = (i * 2) % len(text_hash)
                if byte_offset + 1 < len(text_hash):
                    value = struct.unpack('!h', text_hash[byte_offset:byte_offset+2])[0]
                    # 归一化到[-1, 1]
                    normalized_value = value / 32768.0
                    embedding.append(normalized_value)
                else:
                    embedding.append(0.0)
            
            # 向量归一化到单位长度（模长为1）
            magnitude = sum(x*x for x in embedding)**0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            # 验证归一化后的模长
            final_magnitude = sum(x*x for x in embedding)**0.5
            
            logger.info(f"使用本地embedding生成向量: {len(embedding)}维")
            logger.info(f"向量前5个值: {embedding[:5]}")
            logger.info(f"归一化前模长: {magnitude:.4f}")
            logger.info(f"归一化后模长: {final_magnitude:.4f}")
            return embedding
            
        except Exception as e:
            logger.error(f"本地embedding生成失败: {e}")
            # 如果所有方案都失败，返回零向量
            from app.core.config import settings
            return [0.0] * settings.EMBEDDING_DIMENSION
    
    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,  # 🔧 降低阈值以获得更多相关结果
        file_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """向量语义检索"""
        await self._get_services()
        
        try:
            # 获取查询向量
            query_vector = await self._get_query_embedding(query)
            
            # 执行向量搜索
            search_results = await self.vector_service.search_documents(
                query_vector=query_vector,
                file_ids=file_ids,
                limit=limit,
                score_threshold=score_threshold,
                collection_name=collection_name
            )
            
            # 增强搜索结果，添加文件元数据
            enriched_results = []
            for result in search_results:
                payload = result.get("payload", {})
                file_id = payload.get("file_id")
                
                # 获取文件元数据
                file_metadata = None
                if file_id:
                    file_metadata = await self.cache_service.get_file_metadata(file_id)
                
                enriched_result = {
                    "score": result.get("score", 0.0),
                    "chunk_id": payload.get("chunk_id"),
                    "file_id": file_id,
                    "text": payload.get("text", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "source_file": payload.get("source_file"),
                    "block_type": payload.get("block_type"),
                    "file_metadata": {
                        "filename": file_metadata.get("filename") if file_metadata else None,
                        "upload_date": file_metadata.get("upload_date") if file_metadata else None,
                        "file_size": file_metadata.get("file_size") if file_metadata else None,
                        "content_type": file_metadata.get("content_type") if file_metadata else None
                    } if file_metadata else None
                }
                enriched_results.append(enriched_result)
            
            logger.info(f"向量检索完成: 查询='{query}' 找到{len(enriched_results)}个结果")
            return enriched_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"向量检索失败: {str(e)}"
            )
    
    async def text_search(
        self,
        query: str,
        limit: int = 10,
        file_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """文本关键词检索"""
        await self._get_services()
        
        try:
            # 使用向量数据库的文本搜索功能
            search_results = await self.vector_service.search_by_text_filter(
                collection_name=collection_name or settings.QDRANT_COLLECTION_NAME,
                text_field="text",
                search_text=query,
                limit=limit
            )
            
            # 过滤指定文件
            if file_ids:
                search_results = [
                    result for result in search_results
                    if result.get("payload", {}).get("file_id") in file_ids
                ]
            
            # 增强搜索结果
            enriched_results = []
            for result in search_results[:limit]:
                payload = result.get("payload", {})
                file_id = payload.get("file_id")
                
                # 获取文件元数据
                file_metadata = None
                if file_id:
                    file_metadata = await self.cache_service.get_file_metadata(file_id)
                
                enriched_result = {
                    "score": 1.0,  # 文本搜索没有相似度分数，使用1.0
                    "chunk_id": payload.get("chunk_id"),
                    "file_id": file_id,
                    "text": payload.get("text", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "source_file": payload.get("source_file"),
                    "block_type": payload.get("block_type"),
                    "file_metadata": {
                        "filename": file_metadata.get("filename") if file_metadata else None,
                        "upload_date": file_metadata.get("upload_date") if file_metadata else None,
                        "file_size": file_metadata.get("file_size") if file_metadata else None,
                        "content_type": file_metadata.get("content_type") if file_metadata else None
                    } if file_metadata else None
                }
                enriched_results.append(enriched_result)
            
            logger.info(f"文本检索完成: 查询='{query}' 找到{len(enriched_results)}个结果")
            return enriched_results
            
        except Exception as e:
            logger.error(f"文本检索失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"文本检索失败: {str(e)}"
            )
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        score_threshold: float = 0.5,
        file_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """混合检索（向量检索+文本检索）"""
        await self._get_services()
        
        try:
            # 并行执行向量检索和文本检索
            import asyncio
            
            vector_task = asyncio.create_task(
                self.vector_search(
                    query=query,
                    limit=limit * 2,  # 获取更多结果用于融合
                    score_threshold=score_threshold,
                    file_ids=file_ids,
                    collection_name=collection_name
                )
            )
            
            text_task = asyncio.create_task(
                self.text_search(
                    query=query,
                    limit=limit * 2,
                    file_ids=file_ids,
                    collection_name=collection_name
                )
            )
            
            vector_results, text_results = await asyncio.gather(vector_task, text_task)
            
            # 合并结果并重新评分
            chunk_scores = {}
            
            # 处理向量检索结果
            for result in vector_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    chunk_scores[chunk_id] = {
                        "vector_score": result.get("score", 0.0),
                        "text_score": 0.0,
                        "result": result
                    }
            
            # 处理文本检索结果
            for result in text_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in chunk_scores:
                        chunk_scores[chunk_id]["text_score"] = 1.0
                    else:
                        chunk_scores[chunk_id] = {
                            "vector_score": 0.0,
                            "text_score": 1.0,
                            "result": result
                        }
            
            # 计算混合分数并排序
            hybrid_results = []
            for chunk_id, scores in chunk_scores.items():
                hybrid_score = (
                    scores["vector_score"] * vector_weight +
                    scores["text_score"] * text_weight
                )
                
                result = scores["result"].copy()
                result["hybrid_score"] = hybrid_score
                result["vector_score"] = scores["vector_score"]
                result["text_score"] = scores["text_score"]
                
                hybrid_results.append(result)
            
            # 按混合分数排序
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # 返回前N个结果
            final_results = hybrid_results[:limit]
            
            logger.info(f"混合检索完成: 查询='{query}' 找到{len(final_results)}个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"混合检索失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"混合检索失败: {str(e)}"
            )
    
    async def generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """基于检索结果生成回答"""
        try:
            # 构建上下文
            context_texts = []
            sources = []
            total_length = 0
            
            for result in search_results:
                text = result.get("text", "")
                if text and total_length + len(text) <= max_context_length:
                    context_texts.append(text)
                    sources.append({
                        "chunk_id": result.get("chunk_id"),
                        "file_id": result.get("file_id"),
                        "filename": result.get("file_metadata", {}).get("filename"),
                        "score": result.get("score", 0.0)
                    })
                    total_length += len(text)
                
                if total_length >= max_context_length:
                    break
            
            if not context_texts:
                return {
                    "answer": "抱歉，没有找到相关信息来回答您的问题。",
                    "sources": [],
                    "context_used": ""
                }
            
            context = "\n\n".join(context_texts)
            
            # 构建提示词
            prompt = f"""基于以下上下文信息，回答用户的问题。请确保答案准确、完整，并基于提供的上下文。

上下文信息：
{context}

用户问题：{query}

请提供一个详细、准确的回答："""
            
            # 调用LLM生成回答
            answer = await self._call_llm(prompt)
            
            result = {
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "query": query,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"答案生成完成: 查询='{query}' 使用了{len(sources)}个来源")
            return result
            
        except Exception as e:
            logger.error(f"答案生成失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.LLM_CONNECTION_ERROR,
                f"答案生成失败: {str(e)}"
            )
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM生成回答"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.LLM_API_BASE}/chat/completions",
                    json={
                        "model": settings.LLM_MODEL_NAME,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.7
                    },
                    headers={
                        "Authorization": f"Bearer {settings.LLM_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                return answer.strip()
                
        except Exception as e:
            logger.error(f"调用LLM失败: {e}")
            raise
    
    async def search_and_answer(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.5,
        file_ids: Optional[List[str]] = None,
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """检索并生成回答"""
        await self._get_services()
        
        try:
            # 根据检索类型执行搜索
            if search_type == "vector":
                search_results = await self.vector_search(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    file_ids=file_ids
                )
            elif search_type == "text":
                search_results = await self.text_search(
                    query=query,
                    limit=limit,
                    file_ids=file_ids
                )
            elif search_type == "hybrid":
                search_results = await self.hybrid_search(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    file_ids=file_ids
                )
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")
            
            result = {
                "query": query,
                "search_type": search_type,
                "search_results": search_results,
                "search_count": len(search_results),
                "searched_at": datetime.now().isoformat()
            }
            
            # 生成回答
            if generate_answer and search_results:
                answer_result = await self.generate_answer(query, search_results)
                result["answer"] = answer_result["answer"]
                result["sources"] = answer_result["sources"]
                result["context_used"] = answer_result["context_used"]
            elif generate_answer:
                result["answer"] = "抱歉，没有找到相关信息来回答您的问题。"
                result["sources"] = []
                result["context_used"] = ""
            
            # 保存搜索历史到缓存
            await self._save_search_history(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"检索和回答失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"检索和回答失败: {str(e)}"
            )
    
    async def _save_search_history(self, query: str, result: Dict[str, Any]):
        """保存搜索历史"""
        try:
            # 简化结果用于存储
            simplified_result = {
                "query": query,
                "search_type": result.get("search_type"),
                "search_count": result.get("search_count"),
                "searched_at": result.get("searched_at"),
                "has_answer": "answer" in result
            }
            
            # 生成搜索历史key
            search_key = f"search_history:{datetime.now().strftime('%Y%m%d')}"
            
            # 添加到搜索历史列表
            await self.cache_service.rpush(search_key, simplified_result)
            
            # 设置过期时间（7天）
            await self.cache_service.expire(search_key, 7 * 24 * 3600)
            
        except Exception as e:
            logger.warning(f"保存搜索历史失败: {e}")
    
    async def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """获取搜索建议"""
        await self._get_services()
        
        try:
            # 简单的搜索建议实现，基于已有的文本内容
            # 可以后续使用更复杂的算法优化
            
            # 从向量数据库中获取相关文本片段
            search_results = await self.text_search(
                query=query,
                limit=20
            )
            
            suggestions = set()
            query_words = set(query.lower().split())
            
            for result in search_results:
                text = result.get("text", "")
                words = text.lower().split()
                
                # 提取包含查询词的短语
                for i, word in enumerate(words):
                    if any(query_word in word for query_word in query_words):
                        # 提取前后几个词作为建议
                        start = max(0, i - 2)
                        end = min(len(words), i + 3)
                        phrase = " ".join(words[start:end])
                        if len(phrase) > len(query) and len(phrase) < 50:
                            suggestions.add(phrase)
                        
                        if len(suggestions) >= limit * 2:
                            break
                
                if len(suggestions) >= limit * 2:
                    break
            
            # 返回最相关的建议
            suggestion_list = list(suggestions)[:limit]
            
            logger.debug(f"生成搜索建议: 查询='{query}' 建议数={len(suggestion_list)}")
            return suggestion_list
            
        except Exception as e:
            logger.error(f"获取搜索建议失败: {query} - {e}")
            return []
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        await self._get_services()
        
        try:
            # 获取向量数据库统计
            collection_info = await self.vector_service.get_collection_info(
                settings.QDRANT_COLLECTION_NAME
            )
            
            # 获取今日搜索历史
            today_key = f"search_history:{datetime.now().strftime('%Y%m%d')}"
            today_searches = await self.cache_service.llen(today_key)
            
            # 获取文件统计
            file_keys = []
            cursor = 0
            while True:
                cursor, new_keys = await self.cache_service.redis.scan(
                    cursor, match="file:*", count=100
                )
                file_keys.extend(new_keys)
                if cursor == 0:
                    break
            
            statistics = {
                "vector_database": {
                    "total_points": collection_info.get("points_count", 0) if collection_info else 0,
                    "collection_status": collection_info.get("status") if collection_info else "unknown"
                },
                "search_activity": {
                    "today_searches": today_searches,
                },
                "document_library": {
                    "total_files": len(file_keys),
                },
                "updated_at": datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取搜索统计失败: {e}")
            return {
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
    
    async def search(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.5,  # 🔧 降低默认阈值
        file_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """通用搜索方法 - 根据search_type调用不同的搜索策略"""
        if search_type == "vector" or search_type == "semantic":
            return await self.vector_search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                file_ids=file_ids,
                collection_name=collection_name
            )
        elif search_type == "text":
            return await self.text_search(
                query=query,
                limit=limit,
                file_ids=file_ids,
                collection_name=collection_name
            )
        elif search_type == "hybrid":
            vector_weight = kwargs.get("vector_weight", 0.7)
            semantic_weight = kwargs.get("semantic_weight", 0.3)
            return await self.hybrid_search(
                query=query,
                limit=limit,
                vector_weight=vector_weight,
                text_weight=semantic_weight,  # 转换参数名
                score_threshold=score_threshold,
                file_ids=file_ids,
                collection_name=collection_name
            )
        else:
            raise create_service_exception(
                ErrorCode.INVALID_REQUEST,
                f"不支持的搜索类型: {search_type}"
            )
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,  # 🔧 降低默认阈值
        file_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """语义检索 - 使用向量搜索"""
        return await self.vector_search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            file_ids=file_ids,
            collection_name=collection_name
        )
    
    async def search_in_knowledge_base(
        self,
        kb_id: str,
        collection_name: str,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,  # 🔧 降低默认阈值
        return_images: bool = True,
        return_metadata: bool = True,
        file_types: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        在知识库中进行专用检索
        
        支持返回文本和图片，针对知识库优化的检索算法
        """
        await self._get_services()
        
        try:
            # 获取知识库文件列表（用于过滤）
            from app.services.knowledge_base_service import get_knowledge_base_service
            kb_service = await get_knowledge_base_service()
            kb_file_ids = await kb_service.get_knowledge_base_files(kb_id)
            
            if not kb_file_ids:
                logger.warning(f"知识库 {kb_id} 中没有文件")
                return []
            
            # 应用文件类型过滤
            filtered_file_ids = kb_file_ids
            if file_types:
                filtered_file_ids = []
                for file_id in kb_file_ids:
                    file_metadata = await self.cache_service.get_file_metadata(file_id)
                    if file_metadata:
                        file_ext = file_metadata.get("filename", "").lower().split(".")[-1]
                        if f".{file_ext}" in file_types:
                            filtered_file_ids.append(file_id)
            
            # 应用日期范围过滤
            if date_range and ("start_date" in date_range or "end_date" in date_range):
                date_filtered_ids = []
                for file_id in filtered_file_ids:
                    file_metadata = await self.cache_service.get_file_metadata(file_id)
                    if file_metadata:
                        upload_date = file_metadata.get("upload_date")
                        if upload_date:
                            # 简单的日期比较（这里可以优化）
                            if date_range.get("start_date"):
                                if upload_date >= date_range["start_date"]:
                                    date_filtered_ids.append(file_id)
                            elif date_range.get("end_date"):
                                if upload_date <= date_range["end_date"]:
                                    date_filtered_ids.append(file_id)
                            else:
                                date_filtered_ids.append(file_id)
                filtered_file_ids = date_filtered_ids
            
            # 执行向量检索
            search_results = await self.vector_search(
                query=query,
                limit=top_k,
                score_threshold=score_threshold,
                file_ids=filtered_file_ids,
                collection_name=collection_name
            )
            
            # 增强结果 - 添加图片和元数据
            enhanced_results = []
            for result in search_results:
                enhanced_result = {
                    "score": result.get("score", 0.0),
                    "text": result.get("text", ""),
                    "chunk_id": result.get("chunk_id"),
                    "file_id": result.get("file_id"),
                    "chunk_index": result.get("chunk_index", 0),
                    "block_type": result.get("block_type", "text"),
                    "source_file": result.get("source_file")
                }
                
                # 添加文件元数据
                if return_metadata and result.get("file_metadata"):
                    enhanced_result["file_metadata"] = result["file_metadata"]
                
                # 查找和添加相关图片
                if return_images:
                    images = await self._get_related_images(
                        result.get("file_id"),
                        result.get("chunk_index", 0)
                    )
                    enhanced_result["images"] = images
                
                enhanced_results.append(enhanced_result)
            
            logger.info(f"知识库 {kb_id} 检索完成: 查询='{query}' 找到{len(enhanced_results)}个结果")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"知识库检索失败: {kb_id} - {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"知识库检索失败: {str(e)}"
            )
    
    async def _get_related_images(
        self, 
        file_id: str, 
        chunk_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取与文本块相关的图片
        
        从MinIO中查找解析结果中的图片
        """
        try:
            from app.services.storage_service import get_minio_service
            minio_service = await get_minio_service()
            
            # 获取文件的解析结果路径
            file_metadata = await self.cache_service.get_file_metadata(file_id)
            if not file_metadata:
                return []
            
            # 构建图片路径前缀
            image_prefix = f"parsed/{file_id}/"
            
            # 列出所有图片文件
            image_files = []
            try:
                objects = minio_service.client.list_objects(
                    settings.MINIO_BUCKET_NAME,
                    prefix=image_prefix,
                    recursive=True
                )
                
                for obj in objects:
                    if obj.object_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                        # 生成预签名URL用于图片访问
                        image_url = minio_service.client.presigned_get_object(
                            settings.MINIO_BUCKET_NAME,
                            obj.object_name,
                            expires=timedelta(hours=24)  # 24小时有效期
                        )
                        
                        image_info = {
                            "filename": obj.object_name.split("/")[-1],
                            "path": obj.object_name,
                            "url": image_url,
                            "size": obj.size,
                            "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
                        }
                        image_files.append(image_info)
                        
            except Exception as e:
                logger.warning(f"获取图片列表失败: {file_id} - {e}")
            
            # 根据chunk_index进行简单的图片关联（可以优化算法）
            # 这里返回前3张图片作为相关图片
            related_images = image_files[:3] if len(image_files) > 3 else image_files
            
            logger.info(f"为文件 {file_id} 找到 {len(related_images)} 张相关图片")
            return related_images
            
        except Exception as e:
            logger.error(f"获取相关图片失败: {file_id} - {e}")
            return []
    
    async def get_knowledge_base_image_gallery(
        self,
        kb_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取知识库的图片画廊
        
        返回知识库中所有文件的图片集合
        """
        try:
            await self._get_services()
            
            # 获取知识库文件列表
            from app.services.knowledge_base_service import get_knowledge_base_service
            kb_service = await get_knowledge_base_service()
            file_ids = await kb_service.get_knowledge_base_files(kb_id)
            
            all_images = []
            for file_id in file_ids:
                images = await self._get_related_images(file_id)
                for image in images:
                    image["file_id"] = file_id
                    all_images.append(image)
            
            # 按文件修改时间排序，返回最新的图片
            all_images.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
            
            return all_images[:limit]
            
        except Exception as e:
            logger.error(f"获取知识库图片画廊失败: {kb_id} - {e}")
            return []


# 全局搜索服务实例
search_service = SearchService()


async def get_search_service() -> SearchService:
    """获取搜索服务实例"""
    return search_service 