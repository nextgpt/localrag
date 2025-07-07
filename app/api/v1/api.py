"""
API v1 主路由
组装所有端点路由
"""

from fastapi import APIRouter

from app.api.v1.endpoints import upload, documents, search, tasks, health, knowledge_bases

# 创建v1版本的主路由器
api_router = APIRouter(prefix="/api/v1")

# 注册所有端点路由
api_router.include_router(upload.router)
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"]) 