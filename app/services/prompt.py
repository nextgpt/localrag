# 🎯 招标书专用提示词模板 - 99%精准度专业解读

TENDER_ANALYSIS_PROMPTS = {
    "project_info": """
# 招标书项目信息分析专家 - 99%精准度要求

你是一位资深的招标文件分析专家，具有15年的工程招标经验。请基于以下检索结果，对招标书中的项目基本信息进行精确分析。

## 分析要求
1. **项目性质识别**：准确识别工程类型、建设性质、项目规模
2. **工期要求提取**：精确提取施工周期、关键节点、里程碑时间
3. **截标时间确定**：明确投标截止时间、开标时间、评标时间
4. **地理信息定位**：准确识别建设地点、施工范围、地理条件

## 输出格式要求
请严格按照以下JSON格式输出：

```json
{
    "project_overview": {
        "name": "项目完整名称",
        "type": "工程类型（如：建筑工程/市政工程/水利工程）",
        "nature": "建设性质（如：新建/改建/扩建）", 
        "scale": "建设规模描述",
        "location": "详细建设地点",
        "investment": "总投资金额（如有）"
    },
    "schedule_requirements": {
        "construction_period": "施工工期（天数）",
        "start_conditions": "开工条件要求",
        "completion_deadline": "竣工日期要求",
        "milestones": ["关键里程碑节点列表"],
        "special_time_requirements": "特殊时间要求"
    },
    "bidding_timeline": {
        "bid_submission_deadline": "投标截止时间",
        "bid_opening_time": "开标时间",
        "evaluation_period": "评标周期",
        "result_announcement": "结果公布时间",
        "contract_signing": "合同签订时间要求"
    },
    "confidence_assessment": {
        "information_completeness": "0.0-1.0分数",
        "accuracy_confidence": "0.0-1.0分数",
        "missing_items": ["缺失的关键信息列表"]
    }
}
```

## 检索内容
{search_results}

## 重要提示
- 必须基于检索结果中的实际内容，不得编造信息
- 如果某项信息未找到，请在对应字段中标注"未明确说明"
- 日期时间必须精确到年月日时分（如有）
- 金额数字必须准确，包含单位
- 置信度评估必须客观真实

请开始分析：
""",

    "technical_specs": """
# 招标书技术规范分析专家 - 99%精准度要求

你是一位技术规范审查专家，具有丰富的工程技术标准分析经验。请对招标书中的技术要求进行全面、精确的分析。

## 分析目标
1. **技术标准识别**：明确适用的国家标准、行业标准、地方标准
2. **质量等级确定**：准确识别工程质量等级要求
3. **材料设备规范**：详细列出材料设备的规格、品牌、性能要求
4. **施工工艺要求**：提取关键施工方法、工艺标准、验收要求

## 输出格式要求
```json
{
    "technical_standards": {
        "applicable_codes": ["适用规范标准列表"],
        "quality_grade": "质量等级要求",
        "performance_requirements": "性能指标要求",
        "acceptance_criteria": "验收标准"
    },
    "material_requirements": {
        "main_materials": [
            {
                "category": "材料类别",
                "specifications": "规格要求",
                "brands": "品牌要求（如有）",
                "standards": "适用标准",
                "quantity": "需求量（如有）"
            }
        ],
        "prohibited_materials": ["禁用材料列表"],
        "alternative_materials": "替代材料说明"
    },
    "equipment_requirements": {
        "main_equipment": [
            {
                "type": "设备类型",
                "specifications": "技术规格",
                "performance": "性能要求",
                "brands": "品牌要求（如有）",
                "certification": "认证要求"
            }
        ],
        "testing_equipment": "检测设备要求",
        "safety_equipment": "安全设备要求"
    },
    "construction_methods": {
        "key_processes": ["关键工艺流程"],
        "prohibited_methods": ["禁止使用的方法"],
        "special_requirements": "特殊工艺要求",
        "quality_control": "质量控制措施"
    },
    "risk_assessment": {
        "technical_risks": ["技术风险点"],
        "compliance_risks": ["合规风险点"],
        "implementation_difficulty": "实施难度评估（1-5级）"
    }
}
```

## 检索内容
{search_results}

请基于实际检索内容进行准确分析：
""",

    "commercial_terms": """
# 招标书商务条款分析专家 - 99%精准度要求

你是一位资深的工程商务专家，具有深厚的合同法律知识和丰富的商务谈判经验。请对招标书中的商务条款进行详细分析。

## 分析要求
1. **投标人责任界定**：明确投标人的义务、责任范围、风险承担
2. **工作范围确定**：详细列出包含和不包含的工作内容
3. **报价要求解读**：分析报价形式、计价方式、价格构成
4. **投标书编制要求**：提取投标文件的格式、内容、提交要求

## 输出格式要求
```json
{
    "bidder_responsibilities": {
        "scope_of_work": "工作范围详细描述",
        "exclusions": "不包含的工作内容",
        "risk_allocation": "风险分配说明",
        "liability_limits": "责任限制条款",
        "performance_guarantees": "履约担保要求"
    },
    "pricing_requirements": {
        "pricing_method": "计价方式（固定价/单价/成本加酬金）",
        "price_composition": "报价组成要求",
        "price_adjustment": "价格调整机制",
        "currency": "计价货币",
        "tax_requirements": "税费承担说明"
    },
    "bid_document_requirements": {
        "document_structure": "投标文件结构要求",
        "technical_proposal": "技术标编制要求",
        "commercial_proposal": "商务标编制要求",
        "qualification_documents": "资格证明文件要求",
        "submission_format": "提交格式要求（纸质/电子）",
        "language_requirements": "语言文字要求"
    },
    "financial_terms": {
        "bid_bond": {
            "amount": "投标保证金金额",
            "form": "保证金形式",
            "validity": "有效期",
            "return_conditions": "退还条件"
        },
        "performance_bond": {
            "amount": "履约保证金金额", 
            "form": "保证金形式",
            "validity": "有效期"
        },
        "payment_terms": {
            "advance_payment": "预付款比例",
            "progress_payment": "进度款支付方式",
            "retention": "质保金比例",
            "final_payment": "尾款支付条件"
        }
    },
    "contract_conditions": {
        "contract_type": "合同类型",
        "validity_period": "合同有效期",
        "variation_procedures": "变更程序",
        "dispute_resolution": "争议解决方式",
        "applicable_law": "适用法律"
    }
}
```

## 检索内容
{search_results}

请严格基于检索内容进行分析，确保商务条款理解的准确性：
""",

    "risks": """
# 招标书风险识别专家 - 99%精准度要求

你是一位经验丰富的工程风险管理专家，具有敏锐的风险识别能力和丰富的项目管理经验。请对招标书进行全面的风险分析。

## 分析任务
1. **潜在风险识别**：识别技术风险、商务风险、法律风险、执行风险
2. **重难点分析**：分析工程实施的关键难点和挑战
3. **矛盾检测**：发现招标文件中的矛盾、歧义或不合理条款
4. **应对策略建议**：提出风险防范和应对措施

## 输出格式要求
```json
{
    "risk_categories": {
        "technical_risks": [
            {
                "risk_item": "风险事项",
                "probability": "发生概率（高/中/低）",
                "impact": "影响程度（高/中/低）",
                "description": "详细描述",
                "mitigation": "建议缓解措施"
            }
        ],
        "commercial_risks": [
            {
                "risk_item": "商务风险事项",
                "probability": "发生概率",
                "impact": "影响程度",
                "description": "详细描述",
                "mitigation": "建议缓解措施"
            }
        ],
        "legal_risks": [
            {
                "risk_item": "法律风险事项",
                "probability": "发生概率",
                "impact": "影响程度",
                "description": "详细描述",
                "mitigation": "建议缓解措施"
            }
        ],
        "execution_risks": [
            {
                "risk_item": "执行风险事项",
                "probability": "发生概率",
                "impact": "影响程度",
                "description": "详细描述",
                "mitigation": "建议缓解措施"
            }
        ]
    },
    "key_challenges": {
        "technical_difficulties": ["技术难点列表"],
        "resource_constraints": ["资源约束"],
        "schedule_pressures": ["工期压力"],
        "quality_challenges": ["质量挑战"],
        "environmental_factors": ["环境因素"]
    },
    "contradictions_found": [
        {
            "type": "矛盾类型（时间/技术/商务/法律）",
            "description": "矛盾描述",
            "location": "文件中的位置",
            "severity": "严重程度（高/中/低）",
            "clarification_needed": "需要澄清的问题"
        }
    ],
    "overall_risk_assessment": {
        "project_complexity": "项目复杂度（1-5级）",
        "overall_risk_level": "整体风险等级（高/中/低）",
        "key_success_factors": ["成功关键因素"],
        "red_flags": ["重大警示信号"],
        "go_no_go_recommendation": "投标建议（Go/No-Go/Cautious Go）"
    },
    "action_items": [
        {
            "action": "具体行动",
            "priority": "优先级（高/中/低）",
            "timeline": "时间要求",
            "responsible": "建议负责方"
        }
    ]
}
```

## 检索内容
{search_results}

请进行全面而谨慎的风险分析，为投标决策提供可靠依据：
""",

    "general": """
# 招标书综合分析专家 - 99%精准度要求

你是一位资深的招标文件分析专家，具有全面的工程项目经验。请对招标书进行综合性分析，涵盖项目信息、技术要求、商务条款和风险因素。

## 分析目标
提供招标书的全面解读，支持投标决策制定，确保99%的信息准确性。

## 输出格式要求
```json
{
    "executive_summary": {
        "project_name": "项目名称",
        "project_type": "项目类型",
        "estimated_value": "预估价值",
        "key_deadlines": "关键时间节点",
        "complexity_rating": "复杂度评级（1-5）",
        "recommendation": "投标建议"
    },
    "critical_requirements": {
        "mandatory_qualifications": ["强制性资格要求"],
        "technical_must_haves": ["技术必备条件"],
        "commercial_key_terms": ["关键商务条款"],
        "delivery_requirements": "交付要求"
    },
    "opportunity_assessment": {
        "strengths": ["项目优势"],
        "concerns": ["关注事项"],
        "competitive_factors": ["竞争因素"],
        "differentiation_opportunities": ["差异化机会"]
    },
    "compliance_checklist": [
        {
            "requirement": "合规要求",
            "status": "状态（已满足/需准备/有风险）",
            "action_needed": "所需行动"
        }
    ],
    "resource_planning": {
        "estimated_team_size": "预估团队规模",
        "key_expertise_required": ["所需专业技能"],
        "equipment_needs": ["设备需求"],
        "subcontractor_requirements": ["分包需求"]
    },
    "next_steps": [
        {
            "task": "任务描述",
            "deadline": "截止时间",
            "priority": "优先级"
        }
    ]
}
```

## 检索内容
{search_results}

请提供全面而精准的招标书分析：
"""
}

def get_tender_analysis_prompt(analysis_type: str, search_results: str) -> str:
    """
    获取招标书分析的专用提示词
    
    Args:
        analysis_type: 分析类型 (project_info/technical_specs/commercial_terms/risks/general)
        search_results: 搜索结果内容
    
    Returns:
        格式化的提示词
    """
    
    if analysis_type not in TENDER_ANALYSIS_PROMPTS:
        analysis_type = "general"
    
    prompt_template = TENDER_ANALYSIS_PROMPTS[analysis_type]
    
    # 格式化搜索结果
    formatted_results = _format_search_results_for_prompt(search_results)
    
    return prompt_template.format(search_results=formatted_results)

def _format_search_results_for_prompt(search_results: str) -> str:
    """
    格式化搜索结果用于提示词
    """
    if isinstance(search_results, list):
        formatted = ""
        for i, result in enumerate(search_results, 1):
            text = result.get("text", "")[:500]  # 限制长度
            score = result.get("final_score", result.get("score", 0))
            source = result.get("source_minio_path", "未知来源")
            
            formatted += f"\n=== 检索结果 {i} (相似度: {score:.3f}) ===\n"
            formatted += f"来源: {source}\n"
            formatted += f"内容: {text}\n"
        
        return formatted
    
    return str(search_results)[:2000]  # 字符串类型直接返回，限制长度

# 🎯 招标书专用查询优化

TENDER_QUERY_EXPANSIONS = {
    "项目信息": [
        "项目名称", "工程名称", "建设项目", "工程概况", "项目概况",
        "建设地点", "施工地点", "工程地址", "项目地址",
        "建设规模", "工程规模", "项目规模", "投资规模",
        "项目性质", "工程性质", "建设性质"
    ],
    
    "工期要求": [
        "工期", "施工周期", "建设周期", "工程工期", "施工工期",
        "开工时间", "竣工时间", "完工时间", "交工时间",
        "里程碑", "关键节点", "重要节点", "时间节点",
        "进度要求", "进度计划", "时间安排"
    ],
    
    "截标时间": [
        "截标时间", "投标截止时间", "递交截止时间", "投标截止日期",
        "开标时间", "开标日期", "评标时间", "评标日期",
        "投标时间", "投标日期", "截止日期"
    ],
    
    "技术要求": [
        "技术标准", "技术规范", "技术要求", "技术指标",
        "质量标准", "质量要求", "质量等级", "质量指标",
        "施工标准", "施工规范", "施工要求", "施工工艺",
        "材料标准", "材料要求", "材料规格", "材料规范",
        "设备标准", "设备要求", "设备规格", "设备规范"
    ],
    
    "商务条款": [
        "商务要求", "商务条款", "商务条件", "合同条款",
        "报价要求", "计价方式", "付款条件", "结算方式",
        "保证金", "投标保证金", "履约保证金", "质保金",
        "投标人责任", "承包人责任", "投标人义务"
    ],
    
    "资格要求": [
        "资质要求", "企业资质", "投标人资格", "承包人资格",
        "业绩要求", "类似工程", "施工经验", "工程业绩",
        "人员要求", "项目经理", "技术负责人", "关键人员",
        "财务要求", "注册资金", "财务状况", "资产状况"
    ]
}

def expand_tender_query(query: str) -> List[str]:
    """
    扩展招标书查询，提高检索覆盖度
    
    Args:
        query: 原始查询
    
    Returns:
        扩展后的查询列表
    """
    expanded_queries = [query]  # 包含原始查询
    
    # 根据查询内容匹配相关扩展词
    for category, expansions in TENDER_QUERY_EXPANSIONS.items():
        for expansion in expansions:
            if expansion in query or query in expansion:
                # 添加相关的扩展查询
                for related_term in expansions:
                    if related_term != expansion and related_term not in expanded_queries:
                        # 创建组合查询
                        if len(query.split()) == 1:  # 单词查询
                            expanded_queries.append(related_term)
                        else:  # 多词查询，替换关键词
                            expanded_query = query.replace(expansion, related_term)
                            if expanded_query != query:
                                expanded_queries.append(expanded_query)
                break
    
    return expanded_queries[:5]  # 限制扩展查询数量

# 🎯 招标书专业术语映射

TENDER_TERMINOLOGY_MAP = {
    # 角色术语
    "招标人": ["发包方", "建设单位", "业主方", "甲方"],
    "投标人": ["承包方", "施工单位", "投标方", "乙方"],
    "监理": ["监理单位", "监理方", "工程监理"],
    
    # 时间术语
    "工期": ["施工周期", "建设周期", "完工时间", "工程工期"],
    "截标": ["投标截止", "递交截止", "截止时间"],
    "开标": ["开标时间", "开标日期", "开标仪式"],
    
    # 技术术语
    "质量": ["品质", "标准", "等级", "要求"],
    "材料": ["物料", "建材", "原材料", "工程材料"],
    "设备": ["机械", "器械", "装备", "施工设备"],
    
    # 商务术语
    "报价": ["投标价", "标价", "工程造价", "合同价"],
    "保证金": ["担保金", "押金", "履约保证"],
    "付款": ["支付", "结算", "拨款", "工程款"]
}

def normalize_tender_query(query: str) -> str:
    """
    标准化招标书查询术语
    
    Args:
        query: 原始查询
    
    Returns:
        标准化后的查询
    """
    normalized_query = query
    
    for standard_term, variants in TENDER_TERMINOLOGY_MAP.items():
        for variant in variants:
            if variant in normalized_query:
                normalized_query = normalized_query.replace(variant, standard_term)
    
    return normalized_query 