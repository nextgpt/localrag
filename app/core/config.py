"""
系统配置管理
所有配置项都有明确的默认值，确保系统稳定运行
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
from typing import List, Union, Any
import os


class Settings(BaseSettings):
    """系统配置类 - 所有配置项都有明确的默认值"""
    
    # === 服务器配置 ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # === 数据库配置 ===
    # Qdrant 向量数据库配置
    QDRANT_HOST: str = "192.168.30.54"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""  # 如果需要认证
    QDRANT_COLLECTION_NAME: str = "rag_documents"  # 默认集合名称
    
    # Redis 缓存数据库配置  
    REDIS_HOST: str = "192.168.30.54"
    REDIS_PORT: int = 36379
    REDIS_PASSWORD: str = "8i9o0p-["
    REDIS_DB: int = 0
    
    # === 对象存储配置 ===
    # MinIO 配置
    MINIO_HOST: str = "192.168.30.54"
    MINIO_PORT: int = 19000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "rag-documents"
    MINIO_SECURE: bool = False  # HTTP/HTTPS
    
    # === AI 服务配置 ===
    # SGLang LLM 服务配置
    SGLANG_BASE_URL: str = "http://192.168.30.54:30000"
    SGLANG_API_KEY: str = ""  # 如果需要认证
    SGLANG_MODEL_NAME: str = "default"
    
    # Embedding 服务配置
    EMBEDDING_BASE_URL: str = "http://192.168.30.54:8011/v1"
    EMBEDDING_API_KEY: str = "dummy_key_for_local_service"  # 本地服务不需要认证，设置dummy值
    EMBEDDING_MODEL: str = "Qwen3-Embedding-8B"
    EMBEDDING_DIMENSION: int = 4096  # ⭐ 推荐：使用Qwen3-Embedding-8B的原生维度
    
    # 兼容性别名（为了向后兼容）
    LLM_API_BASE: str = ""  # 会在初始化时同步
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = ""
    EMBEDDING_API_BASE: str = ""  # 会在初始化时同步
    EMBEDDING_MODEL_NAME: str = ""
    SGLANG_API_BASE: str = ""  # 🔧 添加别名，兼容MinerU解析代码
    
    # === 文件处理配置 ===
    # 文件上传限制
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: Any = Field(
        default=[
            ".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls",
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
            ".txt", ".md"
        ],
        description="允许上传的文件扩展名列表"
    )
    
    @field_validator('ALLOWED_EXTENSIONS', mode='before')
    @classmethod
    def validate_allowed_extensions(cls, v) -> List[str]:
        """解析ALLOWED_EXTENSIONS，支持逗号分隔的字符串和列表"""
        default_extensions = [
            ".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls",
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
            ".txt", ".md"
        ]
        
        if isinstance(v, str):
            # 从环境变量读取的逗号分隔字符串
            if not v.strip():
                # 空字符串返回默认值
                return default_extensions
            # 分割并清理空白字符
            extensions = [ext.strip() for ext in v.split(',') if ext.strip()]
            # 确保每个扩展名都以点开头并转为小写
            normalized_extensions = []
            for ext in extensions:
                if not ext.startswith('.'):
                    ext = '.' + ext
                normalized_extensions.append(ext.lower())
            return normalized_extensions
        elif isinstance(v, list):
            # 已经是列表，规范化处理
            normalized_extensions = []
            for ext in v:
                if isinstance(ext, str):
                    if not ext.startswith('.'):
                        ext = '.' + ext
                    normalized_extensions.append(ext.lower())
            return normalized_extensions if normalized_extensions else default_extensions
        else:
            # 其他类型，返回默认值
            return default_extensions
    
    # 存储路径配置
    UPLOAD_PATH: str = "./uploads"
    PARSED_OUTPUT_PATH: str = "./parsed_output"
    RAG_STORAGE_PATH: str = "./rag_storage"
    STATIC_IMAGES_PATH: str = "./static/images"
    TECHNICAL_DOCS_PATH: str = "./technical_docs"
    
    # 路径兼容性别名
    PARSED_OUTPUT_DIR: str = ""  # 会在初始化时同步
    RAG_WORKING_DIR: str = ""  # 会在初始化时同步
    
    # === RAG 系统配置 ===
    # RAGAnything 配置
    RAG_WORKING_DIR: str = "./rag_storage"
    MINERU_PARSE_METHOD: str = "auto"  # auto, ocr, txt
    ENABLE_IMAGE_PROCESSING: bool = True
    ENABLE_TABLE_PROCESSING: bool = True
    ENABLE_EQUATION_PROCESSING: bool = True
    MAX_CONCURRENT_FILES: int = 4
    
    # 检索配置
    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 100
    
    # === 系统配置 ===
    # 任务管理
    TASK_CLEANUP_INTERVAL: int = 3600  # 1小时，清理完成的任务
    TASK_MAX_RETENTION: int = 86400  # 24小时，任务最大保留时间
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/rag-anything.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # 安全配置
    API_KEY_REQUIRED: bool = False  # 是否需要 API Key
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # 每分钟请求数限制
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 同步兼容性别名
        self._sync_compatibility_aliases()
        # 确保必要的目录存在
        self._create_directories()
    
    def _sync_compatibility_aliases(self):
        """同步兼容性别名到主要配置值"""
        # LLM 服务别名同步
        object.__setattr__(self, 'LLM_API_BASE', self.SGLANG_BASE_URL)
        object.__setattr__(self, 'LLM_API_KEY', self.SGLANG_API_KEY)
        object.__setattr__(self, 'LLM_MODEL_NAME', self.SGLANG_MODEL_NAME)
        object.__setattr__(self, 'SGLANG_API_BASE', self.SGLANG_BASE_URL)  # 🔧 同步SGLANG_API_BASE别名
        
        # Embedding 服务别名同步
        object.__setattr__(self, 'EMBEDDING_API_BASE', self.EMBEDDING_BASE_URL)
        object.__setattr__(self, 'EMBEDDING_MODEL_NAME', self.EMBEDDING_MODEL)
        
        # 路径别名同步
        object.__setattr__(self, 'PARSED_OUTPUT_DIR', self.PARSED_OUTPUT_PATH)
        object.__setattr__(self, 'RAG_WORKING_DIR', self.RAG_STORAGE_PATH)
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.UPLOAD_PATH,
            self.PARSED_OUTPUT_PATH,
            self.RAG_STORAGE_PATH,
            self.STATIC_IMAGES_PATH,
            self.TECHNICAL_DOCS_PATH,
            os.path.dirname(self.LOG_FILE)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# 创建全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings 