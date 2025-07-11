#!/usr/bin/env python3
"""
向量相似度调试脚本
分析为什么向量搜索返回0结果
"""

import asyncio
import httpx
import json
import numpy as np
from typing import List, Dict, Any


async def debug_vector_search():
    """调试向量搜索问题"""
    print("🔍 向量搜索调试分析")
    print("=" * 50)
    
    target_file_id = "92d8ab8d-8294-4929-a8d6-9f8dd1285675"
    
    # 1. 获取存储在Qdrant中的实际向量
    print("1️⃣ 获取Qdrant中的实际向量...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 获取一些实际的向量数据
            scroll_payload = {
                "limit": 3,
                "with_vector": True,
                "with_payload": True,
                "filter": {
                    "must": [
                        {
                            "key": "file_id",
                            "match": {
                                "value": target_file_id
                            }
                        }
                    ]
                }
            }
            
            response = await client.post(
                "http://192.168.30.54:6333/collections/rag_documents/points/scroll",
                json=scroll_payload
            )
            
            if response.status_code == 200:
                data = response.json()
                points = data.get("result", {}).get("points", [])
                if points:
                    stored_vector = points[0]["vector"]
                    stored_text = points[0]["payload"].get("text", "")[:200]
                    
                    print(f"   ✅ 获取到存储向量，维度: {len(stored_vector)}")
                    print(f"   📝 文本内容: {stored_text}...")
                    print(f"   🔢 向量前10个值: {stored_vector[:10]}")
                    
                    # 2. 测试不同阈值的搜索效果
                    print("\n2️⃣ 测试不同阈值的搜索效果...")
                    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    
                    for threshold in thresholds:
                        search_payload = {
                            "vector": stored_vector,  # 使用相同的向量
                            "limit": 5,
                            "score_threshold": threshold,
                            "with_payload": True,
                            "filter": {
                                "must": [
                                    {
                                        "key": "file_id",
                                        "match": {
                                            "value": target_file_id
                                        }
                                    }
                                ]
                            }
                        }
                        
                        search_response = await client.post(
                            "http://192.168.30.54:6333/collections/rag_documents/points/search",
                            json=search_payload
                        )
                        
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            results = search_data.get("result", [])
                            scores = [r.get("score", 0) for r in results] if results else []
                            
                            print(f"   阈值 {threshold}: {len(results)}个结果, 最高分: {max(scores) if scores else 'N/A'}")
                        else:
                            print(f"   阈值 {threshold}: 搜索失败 - {search_response.status_code}")
                    
                    # 3. 测试随机向量的相似度
                    print("\n3️⃣ 测试随机向量的相似度...")
                    
                    # 生成一些测试向量
                    test_vectors = {
                        "零向量": [0.0] * 4096,
                        "全1向量": [1.0] * 4096,
                        "随机向量": np.random.normal(0, 0.1, 4096).tolist(),
                        "相似向量": [v + np.random.normal(0, 0.001) for v in stored_vector]
                    }
                    
                    for name, test_vector in test_vectors.items():
                        search_payload = {
                            "vector": test_vector,
                            "limit": 3,
                            "score_threshold": 0.0,  # 无阈值限制
                            "with_payload": False,
                            "filter": {
                                "must": [
                                    {
                                        "key": "file_id", 
                                        "match": {
                                            "value": target_file_id
                                        }
                                    }
                                ]
                            }
                        }
                        
                        search_response = await client.post(
                            "http://192.168.30.54:6333/collections/rag_documents/points/search",
                            json=search_payload
                        )
                        
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            results = search_data.get("result", [])
                            if results:
                                top_score = results[0].get("score", 0)
                                print(f"   {name}: 最高相似度 = {top_score:.4f}")
                            else:
                                print(f"   {name}: 无结果")
                        else:
                            print(f"   {name}: 搜索失败")
                    
                else:
                    print("   ❌ 未找到向量数据")
            else:
                print(f"   ❌ 获取向量失败: {response.status_code}")
                
    except Exception as e:
        print(f"   ❌ 调试失败: {e}")


async def test_api_search_with_debug():
    """测试API搜索并输出调试信息"""
    print("\n4️⃣ 测试API搜索...")
    
    test_queries = ["招标方", "项目名称", "工期", "技术要求"]
    
    for query in test_queries:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 测试向量搜索API
                payload = {
                    "query": query,
                    "limit": 5,
                    "score_threshold": 0.1,
                    "file_ids": ["92d8ab8d-8294-4929-a8d6-9f8dd1285675"]
                }
                
                response = await client.post(
                    "http://localhost:8000/api/v1/search/vector",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result_count = len(data.get("data", {}).get("results", []))
                    print(f"   查询 '{query}': {result_count}个结果")
                else:
                    print(f"   查询 '{query}': API失败 - {response.status_code}")
                    print(f"      错误: {response.text[:200]}")
                    
        except Exception as e:
            print(f"   查询 '{query}': 异常 - {e}")


async def main():
    """主函数"""
    await debug_vector_search()
    await test_api_search_with_debug()
    
    print("\n" + "=" * 50)
    print("🎯 调试建议:")
    print("1. 如果相同向量搜索返回高分数，说明Qdrant正常")
    print("2. 如果API搜索仍返回0结果，检查embedding生成")
    print("3. 如果随机向量都有低分数，检查向量质量")
    print("4. 建议重新生成embedding或使用相同的模型")


if __name__ == "__main__":
    asyncio.run(main()) 