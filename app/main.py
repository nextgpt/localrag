"""
FastAPI 应用主入口文件
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import os

from app.core.config import settings
from app.core.exceptions import setup_exception_handlers
from app.api.v1.api import api_router
from app.services import initialize_services, cleanup_services
from app.workers import start_vectorize_worker, stop_vectorize_worker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-anything")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在启动 RAG-Anything 服务...")
    
    try:
        # 创建必要的目录
        logger.info("创建必要的目录...")
        os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
        os.makedirs(settings.PARSED_OUTPUT_DIR, exist_ok=True)
        os.makedirs(settings.RAG_WORKING_DIR, exist_ok=True)
        logger.info("✓ 目录创建完成")
        
        # 初始化所有服务
        logger.info("初始化分布式服务...")
        await initialize_services()
        logger.info("✓ 所有服务初始化完成")
        
        # 启动后台任务处理器
        logger.info("启动后台任务处理器...")
        await start_vectorize_worker()
        logger.info("✓ 向量化任务处理器已启动")
        
        logger.info(f"🚀 RAG-Anything 服务启动完成！")
        logger.info(f"📊 服务器运行在: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"📚 API 文档: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info(f"🔍 ReDoc 文档: http://{settings.HOST}:{settings.PORT}/redoc")
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭 RAG-Anything 服务...")
    try:
        # 停止后台任务处理器
        logger.info("停止后台任务处理器...")
        await stop_vectorize_worker()
        logger.info("✓ 向量化任务处理器已停止")
        
        # 清理所有服务
        await cleanup_services()
        logger.info("✓ 所有服务清理完成")
    except Exception as e:
        logger.error(f"服务清理失败: {e}")
    
    logger.info("RAG-Anything 服务已关闭")


# 创建 FastAPI 应用实例
app = FastAPI(
    title="RAG-Anything API",
    description="多模态文档处理和检索系统 - 基于分布式存储架构",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 设置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置异常处理
setup_exception_handlers(app)

# 注册路由
app.include_router(api_router)

# 根路径健康检查
@app.get("/", tags=["根目录"])
async def root():
    """根路径接口，返回服务基本信息"""
    return {
        "service": "RAG-Anything API",
        "version": "2.0.0",
        "description": "多模态文档处理和检索系统 - 基于分布式存储架构",
        "status": "running",
        "architecture": {
            "storage": "MinIO (分布式对象存储)",
            "cache": "Redis (缓存和任务管理)",
            "vector_db": "Qdrant (向量数据库)",
            "llm": "SGLang (大语言模型服务)",
            "embedding": "Qwen3-Embedding-8B",
            "parser": "MinerU (文档解析)"
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "upload": "/api/v1/documents/upload",
            "search": "/api/v1/search"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 