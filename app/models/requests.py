"""
API 请求数据模型
定义所有API接口的请求参数结构
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class SearchType(str, Enum):
    """检索类型枚举"""
    VECTOR = "vector"      # 向量检索
    SEMANTIC = "semantic"  # 语义检索
    HYBRID = "hybrid"      # 混合检索


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"    # 等待中
    RUNNING = "running"    # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"      # 失败


class FileUploadRequest(BaseModel):
    """文件上传请求模型"""
    # 注意：实际的文件数据通过 UploadFile 处理，这里定义元数据
    description: Optional[str] = Field(None, max_length=500, description="文件描述")
    tags: Optional[List[str]] = Field(None, max_items=10, description="文件标签")
    auto_parse: bool = Field(True, description="是否自动解析文件")


class SearchRequest(BaseModel):
    """检索请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="检索查询")
    search_type: SearchType = Field(SearchType.HYBRID, description="检索类型")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量限制")
    offset: int = Field(0, ge=0, description="结果偏移量")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="相似度阈值")
    file_ids: Optional[List[str]] = Field(None, description="限制检索的文件ID列表")
    
    @validator('query')
    def validate_query(cls, v):
        """验证查询字符串"""
        if not v.strip():
            raise ValueError("查询不能为空")
        return v.strip()


class FileDeleteRequest(BaseModel):
    """文件删除请求模型"""
    file_ids: List[str] = Field(..., min_items=1, max_items=50, description="要删除的文件ID列表")
    delete_parsed_data: bool = Field(True, description="是否删除解析后的数据")
    delete_vector_data: bool = Field(True, description="是否删除向量数据")


class TaskQueryRequest(BaseModel):
    """任务查询请求模型"""
    status: Optional[TaskStatus] = Field(None, description="任务状态过滤")
    created_by: Optional[str] = Field(None, description="创建者过滤")
    limit: int = Field(10, ge=1, le=100, description="结果数量限制")
    offset: int = Field(0, ge=0, description="结果偏移量")


class DocumentProcessRequest(BaseModel):
    """文档处理请求模型"""
    file_id: str = Field(..., description="文件ID")
    parse_method: str = Field("auto", description="解析方法")
    enable_image_processing: bool = Field(True, description="启用图像处理")
    enable_table_processing: bool = Field(True, description="启用表格处理")
    enable_equation_processing: bool = Field(True, description="启用公式处理")
    
    @validator('parse_method')
    def validate_parse_method(cls, v):
        """验证解析方法"""
        allowed_methods = ["auto", "ocr", "txt"]
        if v not in allowed_methods:
            raise ValueError(f"解析方法必须是: {', '.join(allowed_methods)}")
        return v


class VectorStoreRequest(BaseModel):
    """向量存储请求模型"""
    file_id: str = Field(..., description="文件ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    overwrite: bool = Field(False, description="是否覆盖已有数据")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    server: str
    qdrant: str
    redis: str
    minio: str
    sglang: str
    embedding: str


class BatchFileOperationRequest(BaseModel):
    """批量文件操作请求 - 参考mineru-web的批量处理功能"""
    operation: str = Field(..., description="操作类型：delete/parse/vectorize", pattern="^(delete|parse|vectorize)$")
    file_ids: List[str] = Field(..., description="文件ID列表", min_items=1, max_items=50)
    options: Optional[Dict[str, Any]] = Field(None, description="操作选项")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation": "parse",
                "file_ids": ["file_123", "file_456", "file_789"],
                "options": {
                    "priority": 1,
                    "enable_ocr": True,
                    "extract_images": True
                }
            }
        }


class BatchUploadRequest(BaseModel):
    """批量上传请求 - 类似mineru-web的批量上传功能"""
    files: List[Dict[str, Any]] = Field(..., description="文件信息列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="通用元数据")
    enable_auto_parse: bool = Field(True, description="是否自动启动解析")
    parse_priority: int = Field(0, description="解析优先级")
    
    class Config:
        json_schema_extra = {
            "example": {
                "files": [
                    {
                        "filename": "document1.pdf",
                        "content_type": "application/pdf",
                        "size": 1024000
                    },
                    {
                        "filename": "document2.docx",
                        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "size": 512000
                    }
                ],
                "metadata": {
                    "category": "reports",
                    "department": "research"
                },
                "enable_auto_parse": True,
                "parse_priority": 1
            }
        }


class FilePreviewRequest(BaseModel):
    """文件预览请求"""
    file_id: str = Field(..., description="文件ID")
    preview_type: str = Field("url", description="预览类型：url/thumbnail/content")
    expires: int = Field(3600, description="链接有效期（秒）", ge=60, le=86400)
    options: Optional[Dict[str, Any]] = Field(None, description="预览选项")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file_123",
                "preview_type": "url",
                "expires": 3600,
                "options": {
                    "width": 800,
                    "height": 600,
                    "format": "jpeg"
                }
            }
        }


class TaskManagementRequest(BaseModel):
    """任务管理请求 - 增强任务队列管理"""
    action: str = Field(..., description="操作类型：cancel/retry/priority/pause/resume")
    task_ids: List[str] = Field(..., description="任务ID列表")
    options: Optional[Dict[str, Any]] = Field(None, description="操作选项")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "retry",
                "task_ids": ["task_123", "task_456"],
                "options": {
                    "max_retries": 3,
                    "delay": 60,
                    "priority": 1
                }
            }
        } 