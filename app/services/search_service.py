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

    async def search_tender_documents(
        self,
        query: str,
        file_ids: Optional[List[str]] = None,
        analysis_type: str = "general",
        limit: int = 20,
        score_threshold: float = 0.4,  # 降低阈值，提高召回
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """🎯 招标书专用搜索分析
        
        Args:
            query: 搜索查询
            file_ids: 文件ID列表
            analysis_type: 分析类型 (general/project_info/technical_specs/commercial_terms/risks)
            limit: 结果数量限制
            score_threshold: 相似度阈值
            collection_name: 向量集合名称
        
        Returns:
            结构化的招标书分析结果
        """
        try:
            await self._get_services()
            
            # 1️⃣ 查询预处理和扩展
            enhanced_queries = self._expand_tender_query(query, analysis_type)
            
            # 2️⃣ 多层次检索策略
            all_results = []
            for enhanced_query in enhanced_queries:
                results = await self.vector_search(
                    query=enhanced_query["query"],
                    file_ids=file_ids,
                    limit=limit * 2,  # 增大搜索范围
                    score_threshold=score_threshold,
                    collection_name=collection_name
                )
                
                # 为结果添加查询类型标记
                for result in results:
                    result["query_type"] = enhanced_query["type"]
                    result["query_importance"] = enhanced_query["importance"]
                
                all_results.extend(results)
            
            # 3️⃣ 结果去重和重新排序
            deduped_results = self._deduplicate_tender_results(all_results)
            reranked_results = self._rerank_tender_results(deduped_results, query, analysis_type)
            
            # 4️⃣ 结构化分析
            structured_analysis = await self._analyze_tender_results(
                reranked_results[:limit], 
                query, 
                analysis_type
            )
            
            # 5️⃣ 生成专业报告
            tender_report = self._generate_tender_report(
                structured_analysis, 
                query, 
                analysis_type
            )
            
            logger.info(f"🎯 招标书专用搜索完成: 查询='{query}' 类型={analysis_type} 找到{len(reranked_results)}个结果")
            
            return {
                "query": query,
                "analysis_type": analysis_type,
                "search_results": reranked_results[:limit],
                "structured_analysis": structured_analysis,
                "tender_report": tender_report,
                "total_results": len(reranked_results),
                "search_strategy": {
                    "enhanced_queries": len(enhanced_queries),
                    "score_threshold": score_threshold,
                    "deduplication": len(all_results) - len(deduped_results)
                }
            }
            
        except Exception as e:
            logger.error(f"招标书专用搜索失败: {query} - {e}")
            raise create_service_exception(
                ErrorCode.SEARCH_FAILED,
                f"招标书搜索失败: {str(e)}"
            )
    
    def _expand_tender_query(self, query: str, analysis_type: str) -> List[Dict[str, Any]]:
        """🔍 招标书查询扩展和优化"""
        
        # 基础查询
        enhanced_queries = [{
            "query": query,
            "type": "original",
            "importance": 1.0
        }]
        
        # 根据分析类型扩展查询
        if analysis_type == "project_info":
            # 项目信息相关扩展
            project_expansions = [
                f"{query} 项目概况",
                f"{query} 工程概况", 
                f"{query} 建设规模",
                f"项目性质 {query}",
                f"建设地点 {query}"
            ]
            for exp in project_expansions:
                enhanced_queries.append({
                    "query": exp,
                    "type": "project_context",
                    "importance": 0.8
                })
        
        elif analysis_type == "technical_specs":
            # 技术规范扩展
            tech_expansions = [
                f"{query} 技术要求",
                f"{query} 质量标准",
                f"{query} 施工工艺",
                f"技术规范 {query}",
                f"工程标准 {query}"
            ]
            for exp in tech_expansions:
                enhanced_queries.append({
                    "query": exp,
                    "type": "technical_context",
                    "importance": 0.9
                })
        
        elif analysis_type == "commercial_terms":
            # 商务条款扩展
            commercial_expansions = [
                f"{query} 报价要求",
                f"{query} 付款条件",
                f"{query} 合同条款",
                f"商务要求 {query}",
                f"价格 {query}"
            ]
            for exp in commercial_expansions:
                enhanced_queries.append({
                    "query": exp,
                    "type": "commercial_context", 
                    "importance": 0.85
                })
        
        elif analysis_type == "risks":
            # 风险分析扩展
            risk_expansions = [
                f"{query} 风险",
                f"{query} 难点",
                f"{query} 注意事项",
                f"风险点 {query}",
                f"潜在问题 {query}"
            ]
            for exp in risk_expansions:
                enhanced_queries.append({
                    "query": exp,
                    "type": "risk_context",
                    "importance": 0.7
                })
        
        # 添加同义词扩展
        synonym_map = {
            "招标人": ["发包方", "建设单位", "业主方"],
            "投标人": ["承包方", "施工单位", "投标方"],
            "工期": ["施工周期", "建设周期", "完工时间"],
            "质量": ["品质", "标准", "等级"],
            "材料": ["物料", "建材", "原材料"],
            "设备": ["机械", "器械", "装备"]
        }
        
        for term, synonyms in synonym_map.items():
            if term in query:
                for synonym in synonyms:
                    syn_query = query.replace(term, synonym)
                    enhanced_queries.append({
                        "query": syn_query,
                        "type": "synonym",
                        "importance": 0.6
                    })
        
        return enhanced_queries
    
    def _deduplicate_tender_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重招标书搜索结果"""
        seen_chunks = set()
        deduped_results = []
        
        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                deduped_results.append(result)
        
        return deduped_results
    
    def _rerank_tender_results(self, results: List[Dict[str, Any]], query: str, analysis_type: str) -> List[Dict[str, Any]]:
        """🎯 招标书结果重新排序"""
        
        for result in results:
            original_score = result.get("score", 0.0)
            
            # 1️⃣ 基于块类型的权重调整
            block_type = result.get("block_type", "text")
            type_boost = {
                "key_info_date_info": 0.2,
                "key_info_amount_info": 0.25,
                "key_info_tech_requirement": 0.2,
                "key_info_qualification": 0.15,
                "table": 0.15,
                "section_aligned": 0.1,
                "text": 0.0
            }.get(block_type, 0.0)
            
            # 2️⃣ 基于查询类型的权重调整
            query_type = result.get("query_type", "original")
            query_boost = {
                "original": 0.0,
                "project_context": 0.1 if analysis_type == "project_info" else 0.05,
                "technical_context": 0.15 if analysis_type == "technical_specs" else 0.05,
                "commercial_context": 0.1 if analysis_type == "commercial_terms" else 0.05,
                "risk_context": 0.1 if analysis_type == "risks" else 0.05,
                "synonym": -0.05  # 同义词查询略微降权
            }.get(query_type, 0.0)
            
            # 3️⃣ 基于重要性分数的调整
            importance_score = result.get("tender_info", {}).get("importance_score", 0.5)
            importance_boost = (importance_score - 0.5) * 0.1  # 重要性分数转换为boost
            
            # 4️⃣ 基于结构化数据的加权
            has_structured_data = bool(result.get("tender_info", {}).get("structured_data"))
            structured_boost = 0.05 if has_structured_data else 0.0
            
            # 计算最终分数
            final_score = original_score + type_boost + query_boost + importance_boost + structured_boost
            result["final_score"] = min(1.0, final_score)
            result["score_details"] = {
                "original": original_score,
                "type_boost": type_boost,
                "query_boost": query_boost,
                "importance_boost": importance_boost,
                "structured_boost": structured_boost
            }
        
                # 按最终分数排序
        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results
    
    async def _analyze_tender_results(self, results: List[Dict[str, Any]], query: str, analysis_type: str) -> Dict[str, Any]:
        """🔬 结构化分析招标书搜索结果"""
        
        analysis = {
            "key_information": self._extract_key_information(results),
            "dates_timeline": self._extract_dates_timeline(results),
            "financial_info": self._extract_financial_info(results),
            "technical_requirements": self._extract_technical_requirements(results),
            "qualification_requirements": self._extract_qualification_requirements(results),
            "risks_and_issues": self._identify_risks_and_issues(results),
            "contradictions": self._detect_contradictions(results),
            "completeness_analysis": self._analyze_completeness(results, analysis_type)
        }
        
        return analysis
    
    def _extract_key_information(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取关键信息"""
        key_info = {
            "project_name": [],
            "project_location": [],
            "project_scale": [],
            "construction_period": [],
            "budget": []
        }
        
        for result in results:
            text = result.get("text", "")
            structured_data = result.get("tender_info", {}).get("structured_data", {})
            
            # 项目名称
            if any(keyword in text for keyword in ["项目名称", "工程名称", "工程项目"]):
                key_info["project_name"].append({
                    "text": text[:200],
                    "score": result.get("final_score", 0),
                    "source": result.get("chunk_id")
                })
            
            # 建设地点
            if any(keyword in text for keyword in ["建设地点", "施工地点", "工程地址"]):
                key_info["project_location"].append({
                    "text": text[:200],
                    "score": result.get("final_score", 0),
                    "source": result.get("chunk_id")
                })
            
            # 建设规模
            if any(keyword in text for keyword in ["建设规模", "工程规模", "项目规模"]):
                key_info["project_scale"].append({
                    "text": text[:200],
                    "score": result.get("final_score", 0),
                    "source": result.get("chunk_id")
                })
            
            # 工期信息
            if structured_data.get("dates") or any(keyword in text for keyword in ["工期", "施工周期", "建设周期"]):
                key_info["construction_period"].append({
                    "text": text[:200],
                    "dates": structured_data.get("dates", []),
                    "score": result.get("final_score", 0),
                    "source": result.get("chunk_id")
                })
            
            # 预算信息
            if structured_data.get("amounts") or any(keyword in text for keyword in ["预算", "投资", "限价"]):
                key_info["budget"].append({
                    "text": text[:200],
                    "amounts": structured_data.get("amounts", []),
                    "score": result.get("final_score", 0),
                    "source": result.get("chunk_id")
                })
        
        # 按分数排序并去重
        for category in key_info:
            key_info[category] = sorted(key_info[category], key=lambda x: x["score"], reverse=True)[:3]
        
        return key_info
    
    def _extract_dates_timeline(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取时间线信息"""
        timeline = {
            "bidding_deadline": [],
            "opening_time": [],
            "construction_start": [],
            "construction_end": [],
            "milestones": []
        }
        
        for result in results:
            text = result.get("text", "")
            structured_data = result.get("tender_info", {}).get("structured_data", {})
            
            # 截标时间
            if any(keyword in text for keyword in ["截标时间", "投标截止", "递交截止"]):
                timeline["bidding_deadline"].append({
                    "text": text[:200],
                    "dates": structured_data.get("deadlines", []),
                    "score": result.get("final_score", 0)
                })
            
            # 开标时间  
            if any(keyword in text for keyword in ["开标时间", "开标日期"]):
                timeline["opening_time"].append({
                    "text": text[:200],
                    "dates": structured_data.get("dates", []),
                    "score": result.get("final_score", 0)
                })
            
            # 里程碑节点
            if any(keyword in text for keyword in ["里程碑", "节点", "关键节点", "重要节点"]):
                timeline["milestones"].append({
                    "text": text[:200],
                    "dates": structured_data.get("dates", []),
                    "score": result.get("final_score", 0)
                })
        
        return timeline
    
    def _extract_financial_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取财务信息"""
        financial = {
            "budget_limit": [],
            "bid_bond": [],
            "performance_bond": [],
            "payment_terms": []
        }
        
        for result in results:
            text = result.get("text", "")
            structured_data = result.get("tender_info", {}).get("structured_data", {})
            
            # 预算限价
            if any(keyword in text for keyword in ["投标限价", "预算金额", "控制价"]):
                financial["budget_limit"].append({
                    "text": text[:200],
                    "amounts": structured_data.get("amounts", []),
                    "score": result.get("final_score", 0)
                })
            
            # 投标保证金
            if any(keyword in text for keyword in ["投标保证金", "保证金"]):
                financial["bid_bond"].append({
                    "text": text[:200], 
                    "amounts": structured_data.get("amounts", []),
                    "score": result.get("final_score", 0)
                })
            
            # 履约保证金
            if any(keyword in text for keyword in ["履约保证金", "履约保证"]):
                financial["performance_bond"].append({
                    "text": text[:200],
                    "amounts": structured_data.get("amounts", []),
                    "score": result.get("final_score", 0)
                })
            
            # 付款条件
            if any(keyword in text for keyword in ["付款条件", "付款方式", "结算方式"]):
                financial["payment_terms"].append({
                    "text": text[:200],
                    "score": result.get("final_score", 0)
                })
        
        return financial
    
    def _extract_technical_requirements(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取技术要求"""
        technical = {
            "quality_standards": [],
            "technical_specs": [],
            "materials": [],
            "equipment": [],
            "construction_methods": []
        }
        
        for result in results:
            text = result.get("text", "")
            block_type = result.get("block_type", "")
            
            # 技术要求相关内容
            if block_type == "key_info_tech_requirement" or any(keyword in text for keyword in ["质量标准", "质量等级"]):
                technical["quality_standards"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["技术规范", "技术标准", "技术要求"]):
                technical["technical_specs"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["材料要求", "材料标准", "材料规格"]):
                technical["materials"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["设备要求", "设备规格", "机械设备"]):
                technical["equipment"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["施工方法", "施工工艺", "施工技术"]):
                technical["construction_methods"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
        
        return technical
    
    def _extract_qualification_requirements(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取资格要求"""
        qualification = {
            "company_qualifications": [],
            "personnel_requirements": [],
            "experience_requirements": [],
            "financial_requirements": []
        }
        
        for result in results:
            text = result.get("text", "")
            block_type = result.get("block_type", "")
            
            if block_type == "key_info_qualification" or any(keyword in text for keyword in ["资质要求", "企业资质"]):
                qualification["company_qualifications"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["人员要求", "项目经理", "技术负责人"]):
                qualification["personnel_requirements"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["业绩要求", "类似工程", "施工经验"]):
                qualification["experience_requirements"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
            
            if any(keyword in text for keyword in ["注册资金", "财务状况", "资产状况"]):
                qualification["financial_requirements"].append({
                    "text": text[:300],
                    "score": result.get("final_score", 0)
                })
        
        return qualification
    
    def _identify_risks_and_issues(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别风险和问题点"""
        risks = []
        
        risk_keywords = [
            "风险", "难点", "注意", "特殊要求", "限制", "禁止",
            "严禁", "必须", "不得", "应当", "违约", "罚款", "扣分"
        ]
        
        for result in results:
            text = result.get("text", "")
            risk_score = 0
            
            for keyword in risk_keywords:
                if keyword in text:
                    risk_score += 1
            
            if risk_score > 0:
                risks.append({
                    "text": text[:400],
                    "risk_score": risk_score,
                    "final_score": result.get("final_score", 0),
                    "risk_keywords": [kw for kw in risk_keywords if kw in text]
                })
        
        # 按风险分数排序
        risks.sort(key=lambda x: (x["risk_score"], x["final_score"]), reverse=True)
        return risks[:10]  # 返回前10个风险点
    
    def _detect_contradictions(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测矛盾和不一致"""
        # 这是一个简化的矛盾检测，实际应用中可以更复杂
        contradictions = []
        
        # 检查日期矛盾
        all_dates = []
        for result in results:
            structured_data = result.get("tender_info", {}).get("structured_data", {})
            dates = structured_data.get("dates", [])
            for date in dates:
                all_dates.append({
                    "date": date,
                    "source": result.get("chunk_id"),
                    "text": result.get("text", "")[:100]
                })
        
        # 检查金额矛盾
        all_amounts = []
        for result in results:
            structured_data = result.get("tender_info", {}).get("structured_data", {})
            amounts = structured_data.get("amounts", [])
            for amount in amounts:
                all_amounts.append({
                    "amount": amount,
                    "source": result.get("chunk_id"),
                    "text": result.get("text", "")[:100]
                })
        
        # 检测逻辑可以在这里扩展
        if len(set([d["date"] for d in all_dates])) != len(all_dates):
            contradictions.append({
                "type": "date_inconsistency",
                "description": "发现重复或相互矛盾的日期信息",
                "details": all_dates
            })
        
        return contradictions
    
    def _analyze_completeness(self, results: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """分析信息完整性"""
        
        # 根据分析类型定义必需信息
        required_info = {
            "project_info": ["项目名称", "建设地点", "建设规模", "项目性质"],
            "technical_specs": ["技术标准", "质量要求", "材料规格", "施工方法"],
            "commercial_terms": ["预算限价", "付款条件", "保证金", "合同条款"],
            "risks": ["风险提示", "注意事项", "特殊要求", "违约条款"]
        }
        
        found_info = set()
        missing_info = []
        
        # 检查已找到的信息
        for result in results:
            text = result.get("text", "")
            for category, keywords in required_info.items():
                if analysis_type == "general" or analysis_type == category:
                    for keyword in keywords:
                        if keyword in text:
                            found_info.add(keyword)
        
        # 确定缺失信息
        for category, keywords in required_info.items():
            if analysis_type == "general" or analysis_type == category:
                for keyword in keywords:
                    if keyword not in found_info:
                        missing_info.append(keyword)
        
        total_required = sum(len(keywords) for category, keywords in required_info.items() 
                            if analysis_type == "general" or analysis_type == category)
        
        completeness_score = len(found_info) / total_required if total_required > 0 else 1.0
        
        return {
            "completeness_score": completeness_score,
            "found_information": list(found_info),
            "missing_information": missing_info,
            "coverage_analysis": f"覆盖了 {len(found_info)}/{total_required} 项必需信息"
        }
    
    def _generate_tender_report(self, analysis: Dict[str, Any], query: str, analysis_type: str) -> Dict[str, Any]:
        """🎯 生成招标书专业分析报告"""
        
        report = {
            "executive_summary": self._generate_executive_summary(analysis, query),
            "detailed_findings": self._generate_detailed_findings(analysis, analysis_type),
            "recommendations": self._generate_recommendations(analysis, analysis_type),
            "risk_assessment": self._generate_risk_assessment(analysis),
            "action_items": self._generate_action_items(analysis, analysis_type),
            "confidence_metrics": self._calculate_confidence_metrics(analysis)
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any], query: str) -> str:
        """生成执行摘要"""
        key_info = analysis.get("key_information", {})
        completeness = analysis.get("completeness_analysis", {})
        risks = analysis.get("risks_and_issues", [])
        
        summary_parts = [
            f"针对查询「{query}」的招标书分析结果如下：",
            f"信息完整性评分：{completeness.get('completeness_score', 0):.1%}",
            f"识别风险点：{len(risks)}个",
            f"关键信息覆盖：{completeness.get('coverage_analysis', '未知')}"
        ]
        
        # 添加关键发现
        if key_info.get("project_name"):
            summary_parts.append(f"项目信息：已识别项目名称等基本信息")
        
        if analysis.get("financial_info", {}).get("budget_limit"):
            summary_parts.append(f"财务信息：已识别预算限价等财务条款")
        
        return "\n".join(summary_parts)
    
    def _generate_detailed_findings(self, analysis: Dict[str, Any], analysis_type: str) -> Dict[str, List[str]]:
        """生成详细发现"""
        findings = {
            "positive_findings": [],
            "concerns": [],
            "missing_information": []
        }
        
        # 正面发现
        if analysis.get("key_information", {}).get("project_name"):
            findings["positive_findings"].append("✅ 项目基本信息完整")
        
        if analysis.get("dates_timeline", {}).get("bidding_deadline"):
            findings["positive_findings"].append("✅ 关键时间节点明确")
        
        if analysis.get("financial_info", {}).get("budget_limit"):
            findings["positive_findings"].append("✅ 财务条款清晰")
        
        # 关注点
        risks = analysis.get("risks_and_issues", [])
        if len(risks) > 5:
            findings["concerns"].append(f"⚠️ 识别到{len(risks)}个潜在风险点，需要重点关注")
        
        contradictions = analysis.get("contradictions", [])
        if contradictions:
            findings["concerns"].append(f"⚠️ 发现{len(contradictions)}处信息不一致，需要澄清")
        
        # 缺失信息
        missing = analysis.get("completeness_analysis", {}).get("missing_information", [])
        for item in missing[:5]:  # 只显示前5个
            findings["missing_information"].append(f"❌ 缺少{item}相关信息")
        
        return findings
    
    def _generate_recommendations(self, analysis: Dict[str, Any], analysis_type: str) -> List[str]:
        """生成建议"""
        recommendations = []
        
        completeness_score = analysis.get("completeness_analysis", {}).get("completeness_score", 0)
        
        if completeness_score < 0.7:
            recommendations.append("🔍 建议进一步收集缺失的关键信息")
        
        risks = analysis.get("risks_and_issues", [])
        if risks:
            recommendations.append("⚠️ 建议针对识别的风险点制定应对措施")
        
        contradictions = analysis.get("contradictions", [])
        if contradictions:
            recommendations.append("📞 建议联系招标方澄清矛盾信息")
        
        if analysis_type == "technical_specs":
            tech_req = analysis.get("technical_requirements", {})
            if not tech_req.get("quality_standards"):
                recommendations.append("🏗️ 建议明确质量标准要求")
        
        return recommendations
    
    def _generate_risk_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险评估"""
        risks = analysis.get("risks_and_issues", [])
        
        risk_levels = {"高": 0, "中": 0, "低": 0}
        
        for risk in risks:
            risk_score = risk.get("risk_score", 0)
            if risk_score >= 3:
                risk_levels["高"] += 1
            elif risk_score >= 2:
                risk_levels["中"] += 1
            else:
                risk_levels["低"] += 1
        
        total_risks = sum(risk_levels.values())
        overall_risk = "低"
        if risk_levels["高"] > 0:
            overall_risk = "高"
        elif risk_levels["中"] > 2:
            overall_risk = "中"
        
        return {
            "overall_risk_level": overall_risk,
            "risk_distribution": risk_levels,
            "total_risks": total_risks,
            "top_risks": risks[:3] if risks else []
        }
    
    def _generate_action_items(self, analysis: Dict[str, Any], analysis_type: str) -> List[Dict[str, Any]]:
        """生成行动项"""
        actions = []
        
        # 基于缺失信息生成行动项
        missing = analysis.get("completeness_analysis", {}).get("missing_information", [])
        for item in missing[:3]:
            actions.append({
                "action": f"收集{item}相关信息",
                "priority": "高",
                "category": "信息收集"
            })
        
        # 基于风险生成行动项
        risks = analysis.get("risks_and_issues", [])
        for risk in risks[:2]:
            actions.append({
                "action": f"分析风险：{risk.get('text', '')[:50]}...",
                "priority": "中",
                "category": "风险分析"
            })
        
        return actions
    
    def _calculate_confidence_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """计算置信度指标"""
        
        completeness_score = analysis.get("completeness_analysis", {}).get("completeness_score", 0)
        risks_count = len(analysis.get("risks_and_issues", []))
        contradictions_count = len(analysis.get("contradictions", []))
        
        # 信息完整性置信度
        info_confidence = completeness_score
        
        # 风险识别置信度（风险点越多，置信度越高）
        risk_confidence = min(1.0, risks_count / 10)
        
        # 一致性置信度（矛盾越少，置信度越高）
        consistency_confidence = max(0.0, 1.0 - contradictions_count * 0.2)
        
        # 整体置信度
        overall_confidence = (info_confidence + risk_confidence + consistency_confidence) / 3
        
        return {
            "overall_confidence": overall_confidence,
            "information_completeness": info_confidence,
            "risk_identification": risk_confidence,
            "consistency_check": consistency_confidence
        }


# 全局搜索服务实例
search_service = SearchService()


async def get_search_service() -> SearchService:
    """获取搜索服务实例"""
    return search_service 