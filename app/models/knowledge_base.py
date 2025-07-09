"""
知识库管理数据模型
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class KnowledgeBaseStatus(str, Enum):
    """知识库状态"""
    ACTIVE = "active"           # 活跃
    INACTIVE = "inactive"       # 非活跃
    INDEXING = "indexing"      # 索引中
    ERROR = "error"            # 错误状态


class QdrantConfig(BaseModel):
    """Qdrant向量数据库配置"""
    collection_name: str = Field(..., description="集合名称")
    vector_size: int = Field(4096, description="向量维度")  # 🔧 更新为Qwen3-Embedding-8B真实维度
    distance_metric: str = Field("Cosine", description="距离度量：Cosine/Euclidean/Dot")
    
    # 检索参数配置
    top_k: int = Field(10, ge=1, le=100, description="返回结果数量")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="相似度阈值")
    
    # HNSW配置（影响检索性能和准确率）
    hnsw_m: int = Field(16, ge=4, le=64, description="HNSW M参数，影响连接数")
    hnsw_ef_construct: int = Field(100, ge=10, le=500, description="构建时搜索深度")
    hnsw_ef_search: int = Field(100, ge=10, le=500, description="搜索时深度")
    
    # 优化配置
    optimizer_deleted_threshold: float = Field(0.2, ge=0.0, le=1.0, description="删除阈值")
    optimizer_vacuum_min_vector_number: int = Field(1000, ge=100, description="最小向量数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "knowledge_base_001",
                "vector_size": 4096,  # 🔧 更新为Qwen3-Embedding-8B真实维度
                "distance_metric": "Cosine",
                "top_k": 10,
                "score_threshold": 0.5,
                "hnsw_m": 16,
                "hnsw_ef_construct": 100,
                "hnsw_ef_search": 100,
                "optimizer_deleted_threshold": 0.2,
                "optimizer_vacuum_min_vector_number": 1000
            }
        }


class KnowledgeBase(BaseModel):
    """知识库模型"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")
    
    # 状态信息
    status: KnowledgeBaseStatus = Field(KnowledgeBaseStatus.ACTIVE, description="知识库状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    # 配置信息
    qdrant_config: QdrantConfig = Field(..., description="Qdrant配置")
    
    # 统计信息
    file_count: int = Field(0, ge=0, description="文件数量")
    document_count: int = Field(0, ge=0, description="文档数量") 
    vector_count: int = Field(0, ge=0, description="向量数量")
    total_size: int = Field(0, ge=0, description="总大小（字节）")
    
    # 标签和分类
    tags: Optional[List[str]] = Field(None, max_items=20, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    
    @validator('name')
    def validate_name(cls, v):
        """验证知识库名称"""
        if not v.strip():
            raise ValueError("知识库名称不能为空")
        return v.strip()


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")
    tags: Optional[List[str]] = Field(None, max_items=20, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    
    # Qdrant配置（可选，使用默认值）
    vector_size: int = Field(4096, description="向量维度")  # 🔧 更新为Qwen3-Embedding-8B真实维度
    distance_metric: str = Field("Cosine", description="距离度量")
    top_k: int = Field(10, ge=1, le=100, description="默认返回结果数")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="默认相似度阈值")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")
    status: Optional[KnowledgeBaseStatus] = Field(None, description="知识库状态")
    tags: Optional[List[str]] = Field(None, max_items=20, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    
    # Qdrant配置更新
    top_k: Optional[int] = Field(None, ge=1, le=100, description="返回结果数")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    hnsw_ef_search: Optional[int] = Field(None, ge=10, le=500, description="搜索深度")


class KnowledgeBaseSearch(BaseModel):
    """知识库检索请求"""
    kb_id: str = Field(..., description="知识库ID")
    query: str = Field(..., min_length=1, max_length=1000, description="检索查询")
    
    # 检索参数（覆盖知识库默认配置）
    top_k: Optional[int] = Field(None, ge=1, le=100, description="返回结果数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    
    # 结果类型
    return_images: bool = Field(True, description="是否返回图片")
    return_metadata: bool = Field(True, description="是否返回元数据")
    
    # 过滤条件
    file_types: Optional[List[str]] = Field(None, description="文件类型过滤")
    date_range: Optional[Dict[str, str]] = Field(None, description="日期范围过滤")


class KnowledgeBaseStats(BaseModel):
    """知识库统计信息"""
    kb_id: str
    name: str
    status: KnowledgeBaseStatus
    
    # 数量统计
    file_count: int
    document_count: int
    vector_count: int
    
    # 大小统计
    total_size: int
    avg_file_size: float
    
    # 类型分布
    file_type_distribution: Dict[str, int]
    
    # 处理状态分布
    parse_status_distribution: Dict[str, int]
    vector_status_distribution: Dict[str, int]
    
    # 时间信息
    created_at: datetime
    last_updated: datetime
    last_indexed: Optional[datetime] 