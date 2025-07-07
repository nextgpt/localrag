# RAG-Anything API 文档

## 概述

RAG-Anything API 是一个基于 FastAPI 构建的多模态文档处理和检索系统，提供完整的文档处理链路：从文件上传、解析、向量化到智能检索和问答生成。

## 快速开始

### 1. 环境配置

```bash
# 复制环境配置文件
cp .env.example .env

# 根据实际环境修改配置
vim .env
```

### 2. 启动服务

```bash
# 方式一：使用启动脚本（推荐）
python start_server.py --install-deps

# 方式二：直接使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 核心API接口

### 1. 文件上传接口

**POST** `/api/v1/upload/file`

上传单个文件并可选择自动解析。

```bash
curl -X POST "http://localhost:8000/api/v1/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "description=测试文档" \
  -F "auto_parse=true"
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "file_id": "uuid-string",
    "original_name": "document.pdf",
    "file_size": 1024000,
    "status": "uploaded",
    "auto_parse": true,
    "parse_task_id": "task-uuid"
  },
  "message": "文件上传成功"
}
```

### 2. 文档解析接口

**POST** `/api/v1/documents/parse`

使用MinerU解析文档内容，支持图像、表格、公式处理。

```bash
curl -X POST "http://localhost:8000/api/v1/documents/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "uuid-string",
    "parse_method": "auto",
    "enable_image_processing": true,
    "enable_table_processing": true,
    "enable_equation_processing": true
  }'
```

### 3. 向量索引接口

**POST** `/api/v1/documents/index`

将解析后的文档内容向量化并存储到向量数据库。

```bash
curl -X POST "http://localhost:8000/api/v1/documents/index" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "uuid-string",
    "collection_name": "my_collection",
    "overwrite": false
  }'
```

### 4. 统一检索接口

**POST** `/api/v1/search/`

支持向量检索、语义检索和混合检索。

```bash
curl -X POST "http://localhost:8000/api/v1/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "search_type": "hybrid",
    "limit": 10,
    "offset": 0,
    "file_ids": ["uuid-string"]
  }'
```

### 5. 问答生成接口

**POST** `/api/v1/search/answer`

基于检索结果生成自然语言答案。

```bash
curl -X POST "http://localhost:8000/api/v1/search/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能的发展历程是什么？",
    "search_type": "hybrid",
    "limit": 5
  }'
```

### 6. 任务管理接口

**GET** `/api/v1/tasks/{task_id}`

查询异步任务的执行状态和结果。

```bash
curl -X GET "http://localhost:8000/api/v1/tasks/task-uuid"
```

## 完整工作流程

### 1. 上传并处理文档

```python
import requests

# 1. 上传文件
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/upload/file",
        files={"file": f},
        data={"auto_parse": True}
    )
    
upload_result = response.json()
file_id = upload_result["data"]["file_id"]
parse_task_id = upload_result["data"]["parse_task_id"]

# 2. 等待解析完成
import time
while True:
    response = requests.get(f"http://localhost:8000/api/v1/tasks/{parse_task_id}")
    task_status = response.json()["data"]
    
    if task_status["status"] == "completed":
        break
    elif task_status["status"] == "failed":
        print("解析失败:", task_status["error"])
        break
    
    time.sleep(2)

# 3. 索引文档
response = requests.post(
    "http://localhost:8000/api/v1/documents/index",
    json={"file_id": file_id}
)
index_task_id = response.json()["data"]["task_id"]

# 4. 等待索引完成
while True:
    response = requests.get(f"http://localhost:8000/api/v1/tasks/{index_task_id}")
    task_status = response.json()["data"]
    
    if task_status["status"] == "completed":
        break
    elif task_status["status"] == "failed":
        print("索引失败:", task_status["error"])
        break
    
    time.sleep(2)
```

### 2. 执行检索和问答

```python
# 检索相关内容
response = requests.post(
    "http://localhost:8000/api/v1/search/",
    json={
        "query": "文档的主要内容是什么？",
        "search_type": "hybrid",
        "limit": 5,
        "file_ids": [file_id]
    }
)
search_results = response.json()

# 生成答案
response = requests.post(
    "http://localhost:8000/api/v1/search/answer",
    json={
        "query": "请总结文档的主要内容",
        "search_type": "hybrid",
        "limit": 5,
        "file_ids": [file_id]
    }
)
answer = response.json()["data"]["answer"]
print("AI回答:", answer)
```

## 系统监控

### 健康检查

```bash
# 完整健康检查
curl http://localhost:8000/api/v1/health/

# 快速健康检查
curl http://localhost:8000/api/v1/health/quick

# 单个服务检查
curl http://localhost:8000/api/v1/health/services/qdrant
```

### 任务统计

```bash
# 获取任务统计信息
curl http://localhost:8000/api/v1/tasks/stats/summary

# 获取检索统计信息
curl http://localhost:8000/api/v1/search/stats
```

## 错误处理

API 使用统一的错误响应格式：

```json
{
  "success": false,
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "文件大小超过限制",
    "details": {
      "max_size": 104857600,
      "actual_size": 200000000
    }
  },
  "request_id": "req-uuid",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 限制说明

- 单个文件最大 100MB
- 批量上传最多 10 个文件
- 单次检索最多返回 100 个结果
- 支持的文件格式：PDF、Word、PowerPoint、Excel、图像、文本等

## 技术支持

如有问题请参考：
1. API 文档: http://localhost:8000/docs
2. 系统健康检查: http://localhost:8000/api/v1/health
3. 查看服务器日志获取详细错误信息 