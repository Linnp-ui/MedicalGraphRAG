import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from .query_expander import MedicalQueryExpander
from ..core.community_detector import get_community_detector
from ..core.summary_generator import get_summary_generator
from ..core.cache import cached, get_query_cache
from ..utils.process_monitor import track_process, get_structured_logger

_structured_logger = get_structured_logger("retrieval")


class DRIFTSearch:
    """DRIFT Search - 动态检索策略选择器
    
    根据查询意图动态选择：
    - 全局搜索（Global Search）：适用于总结性、概括性问题
    - 局部搜索（Local Search）：适用于特定实体的精细化推理
    - 混合搜索（Hybrid Search）：结合向量和图谱检索
    """

    def __init__(
        self,
        vector_top_k: int = 5,
        graph_top_k: int = 10,
        enable_cache: bool = True,
        enable_query_expansion: bool = True,
    ):
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.enable_cache = enable_cache
        self.enable_query_expansion = enable_query_expansion
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self.community_detector = get_community_detector()
        self.summary_generator = get_summary_generator()
        self.query_expander = MedicalQueryExpander()

    def _classify_query_intent(self, query: str) -> str:
        """分类查询意图
        
        返回:
            - 'global': 全局/总结性查询
            - 'local': 局部/实体查询
            - 'hybrid': 混合查询
        """
        global_indicators = [
            "总结", "概述", "整体", "主要", "全部", "共有",
            "介绍", "简述", "总览", "概况", "全面", "整体来看",
            "对比", "比较", "区别", "异同", "分类",
        ]
        
        local_indicators = [
            "谁", "什么", "哪个", "详细", "具体", "如何",
            "有哪些", "是什么", "在哪里", "为什么", "怎么样",
            "原因", "治疗", "症状", "副作用", "用法", "用量",
            "检查", "预防", "注意", "建议",
        ]
        
        numeric_indicators = [
            "多少", "数量", "统计", "排名", "前", "最高", "最低",
        ]
        
        query_lower = query.lower()
        
        global_count = sum(1 for ind in global_indicators if ind in query_lower)
        local_count = sum(1 for ind in local_indicators if ind in query_lower)
        numeric_count = sum(1 for ind in numeric_indicators if ind in query_lower)
        
        if global_count > local_count:
            return "global"
        elif local_count > global_count:
            return "local"
        elif numeric_count > 0:
            return "local"
        elif local_count == global_count and local_count > 0:
            # 边界情况：关键词计数相同时，优先走 local（更常见的查询模式）
            return "local"
        else:
            return "hybrid"

    @cached(get_query_cache)
    @track_process("retrieval.global_search")
    def global_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """全局搜索 - 使用社区摘要回答总结性问题"""
        logger.info(f"Executing global search for: {query}")
        
        _structured_logger.info(
            "global_search_started",
            query_length=len(query),
            top_k=top_k,
        )
        
        global_summary = self.summary_generator.generate_global_summary()
        
        top_communities = self.community_detector.get_top_communities(top_n=top_k)
        community_summaries = {}
        
        for comm_id, count in top_communities:
            community_summaries[comm_id] = self.summary_generator.generate_community_summary(comm_id)
        
        result = {
            "search_type": "global",
            "query": query,
            "global_summary": global_summary,
            "community_summaries": community_summaries,
            "top_communities": top_communities,
        }
        
        _structured_logger.info(
            "global_search_completed",
            community_count=len(community_summaries),
        )
        
        return result

    @cached(get_query_cache)
    @track_process("retrieval.local_search")
    def local_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """局部搜索 - 从实体出发进行精细化推理"""
        logger.info(f"Executing local search for: {query}")
        
        _structured_logger.info(
            "local_search_started",
            query_length=len(query),
            top_k=top_k,
        )
        
        entities = self.graph_retriever.find_entities_by_embedding(query, limit=top_k)
        
        if not entities:
            entities = self.graph_retriever.find_entities(entity_name=query, limit=top_k)
        
        results = []
        for entity in entities:
            entity_name = entity.get("name")
            relationships = self.graph_retriever.find_relationships(entity_name, depth=2, limit=5)
            summary = self.summary_generator.generate_entity_summary(entity_name)
            
            results.append({
                "entity": entity_name,
                "type": entity.get("type"),
                "score": entity.get("score", 1.0),
                "summary": summary,
                "relationships": relationships,
            })
        
        result = {
            "search_type": "local",
            "query": query,
            "results": results,
            "entity_count": len(results),
        }
        
        _structured_logger.info(
            "local_search_completed",
            entity_count=len(results),
            found_entities=bool(entities),
        )
        
        return result

    @track_process("retrieval.local_search_async")
    async def local_search_async(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """局部搜索 - 异步版本"""
        logger.info(f"Executing async local search for: {query}")
        
        _structured_logger.info(
            "local_search_async_started",
            query_length=len(query),
            top_k=top_k,
        )
        
        entities = await self.graph_retriever.find_entities_by_embedding_async(query, limit=top_k)
        
        if not entities:
            entities = await self.graph_retriever.find_entities_async(entity_name=query, limit=top_k)
        
        results = []
        for entity in entities:
            entity_name = entity.get("name")
            if not entity_name:
                continue
            relationships = await self.graph_retriever.find_relationships_async(entity_name, depth=2, limit=5)
            summary = self.summary_generator.generate_entity_summary(entity_name)
            
            results.append({
                "entity": entity_name,
                "type": entity.get("type"),
                "score": entity.get("score", 1.0),
                "summary": summary,
                "relationships": relationships,
            })
        
        result = {
            "search_type": "local",
            "query": query,
            "results": results,
            "entity_count": len(results),
        }
        
        _structured_logger.info(
            "local_search_async_completed",
            entity_count=len(results),
            found_entities=bool(entities),
        )
        
        return result

    @cached(get_query_cache)
    @track_process("retrieval.hybrid_search")
    def hybrid_search(self, query: str, alpha: Optional[float] = None) -> Dict[str, Any]:
        """混合搜索 - 结合向量和图谱检索"""
        logger.info(f"Executing hybrid search for: {query}")
        
        if alpha is None:
            alpha = self._compute_dynamic_alpha(query)
        
        _structured_logger.info(
            "hybrid_search_started",
            query_length=len(query),
            alpha=alpha,
        )
        
        vector_results = self.vector_retriever.search(query, top_k=self.vector_top_k)
        graph_results = self.graph_retriever.find_entities_by_embedding(query, limit=self.graph_top_k)
        
        combined_results = self._combine_results(query, vector_results, graph_results, alpha)
        
        result = {
            "search_type": "hybrid",
            "query": query,
            "alpha": alpha,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_results": combined_results,
        }
        
        _structured_logger.info(
            "hybrid_search_completed",
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            combined_count=len(combined_results),
        )
        
        return result

    @track_process("retrieval.hybrid_search_async")
    async def hybrid_search_async(self, query: str, alpha: Optional[float] = None) -> Dict[str, Any]:
        """混合搜索 - 异步版本"""
        logger.info(f"Executing async hybrid search for: {query}")
        
        if alpha is None:
            alpha = self._compute_dynamic_alpha(query)
        
        _structured_logger.info(
            "hybrid_search_async_started",
            query_length=len(query),
            alpha=alpha,
        )
        
        # 并行执行向量和图谱检索
        vector_task = asyncio.create_task(self.vector_retriever.search_async(query, top_k=self.vector_top_k))
        graph_task = asyncio.create_task(self.graph_retriever.find_entities_by_embedding_async(query, limit=self.graph_top_k))
        
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        combined_results = self._combine_results(query, vector_results, graph_results, alpha)
        
        result = {
            "search_type": "hybrid",
            "query": query,
            "alpha": alpha,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_results": combined_results,
        }
        
        _structured_logger.info(
            "hybrid_search_async_completed",
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            combined_count=len(combined_results),
        )
        
        return result

    def _compute_dynamic_alpha(self, query: str) -> float:
        """动态计算alpha权重"""
        intent = self._classify_query_intent(query)
        
        if intent == "global":
            return 0.2
        elif intent == "local":
            return 0.7
        else:
            return 0.5

    def _combine_results(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """组合向量和图谱检索结果"""
        combined = []
        seen = set()
        
        vector_max_score = max((r.get("similarity", 0) for r in vector_results), default=1.0)
        
        for result in vector_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                norm_score = result.get("similarity", 0) / (vector_max_score if vector_max_score > 0 else 1)
                combined.append({
                    **result,
                    "source": "vector",
                    "combined_score": norm_score * alpha,
                })
        
        for entity in graph_results:
            entity_name = entity.get("name")
            if entity_name and entity_name not in seen:
                seen.add(entity_name)
                combined.append({
                    "entity_name": entity_name,
                    "entity_type": entity.get("type"),
                    "score": entity.get("score", 1.0),
                    "source": "graph",
                    "combined_score": entity.get("score", 1.0) * (1 - alpha),
                })
        
        combined.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return combined[:self.vector_top_k + self.graph_top_k]

    @cached(get_query_cache)
    def search(self, query: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """执行DRIFT搜索 - 根据查询自动选择策略，支持查询扩展"""
        if strategy:
            search_strategy = strategy
        else:
            search_strategy = self._classify_query_intent(query)

        logger.info(f"Query: '{query}', Strategy: {search_strategy}")

        # 执行主查询检索
        if search_strategy == "global":
            result = self.global_search(query)
        elif search_strategy == "local":
            result = self.local_search(query)
        else:
            result = self.hybrid_search(query)

        # 查询扩展：用同义词变体补充检索，合并去重
        if self.enable_query_expansion:
            result = self._expand_and_merge(query, result, search_strategy)

        return result

    def _expand_and_merge(self, query: str, main_result: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """用查询扩展变体补充检索，合并去重结果"""
        # 获取意图分类（细粒度，用于 query_expander）
        intent = self._classify_fine_intent(query)
        variants = self.query_expander.expand(query, intent=intent)

        # 过滤掉与原查询相同的变体
        extra_variants = [v for v in variants if v != query]
        if not extra_variants:
            return main_result

        logger.info(f"Query expansion: {len(extra_variants)} variants for '{query}'")

        # 收集已有实体名，用于去重
        existing_entities = set()
        for r in main_result.get("results", []):
            name = r.get("entity") or r.get("name", "")
            if name:
                existing_entities.add(name.lower())

        # 对每个变体执行检索，合并新结果
        merged_results = list(main_result.get("results", []))
        for variant in extra_variants[:3]:  # 最多3个变体，避免过多请求
            try:
                if strategy == "global":
                    variant_result = self.global_search(variant)
                elif strategy == "local":
                    variant_result = self.local_search(variant)
                else:
                    variant_result = self.hybrid_search(variant)

                for r in variant_result.get("results", []):
                    name = r.get("entity") or r.get("name", "")
                    if name and name.lower() not in existing_entities:
                        existing_entities.add(name.lower())
                        # 标记来源为扩展查询
                        r["_expanded_from"] = variant
                        merged_results.append(r)
            except Exception as e:
                logger.warning(f"Query expansion variant '{variant}' failed: {e}")

        main_result["results"] = merged_results
        main_result["expanded_queries"] = extra_variants[:3]
        _structured_logger.info(
            "query_expansion_completed",
            original_query=query,
            variant_count=len(extra_variants),
            merged_result_count=len(merged_results),
        )
        return main_result

    def _classify_fine_intent(self, query: str) -> str:
        """细粒度意图分类，用于查询扩展模板选择"""
        query_lower = query.lower()
        # 药物相关
        drug_indicators = ["药", "素", "苷", "林", "平", "汀", "普利", "沙坦", "洛尔", "阿司匹林", "胰岛素", "二甲双胍"]
        if any(ind in query for ind in drug_indicators):
            if "相互作用" in query or "联用" in query:
                return "drug_interaction"
            return "drug_query"
        # 诊断相关
        if any(ind in query for ind in ["诊断", "鉴别", "检查", "提示"]):
            return "diagnosis_assist"
        # 治疗相关
        if any(ind in query for ind in ["治疗", "用药", "手术", "方案"]):
            return "treatment_query"
        # 检查相关
        if any(ind in query for ind in ["检查", "检验", "指标", "正常值"]):
            return "examination_query"
        # 症状相关
        if any(ind in query for ind in ["症状", "表现", "感觉"]):
            return "symptom_query"
        return "general"

    @track_process("retrieval.search_async")
    async def search_async(self, query: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """执行DRIFT搜索 - 异步版本，支持查询扩展"""
        if strategy:
            search_strategy = strategy
        else:
            search_strategy = self._classify_query_intent(query)

        logger.info(f"Async Query: '{query}', Strategy: {search_strategy}")

        # 执行主查询检索
        if search_strategy == "global":
            result = self.global_search(query)
        elif search_strategy == "local":
            result = await self.local_search_async(query)
        else:
            result = await self.hybrid_search_async(query)

        # 查询扩展：用同义词变体补充检索，合并去重
        if self.enable_query_expansion:
            result = await self._expand_and_merge_async(query, result, search_strategy)

        return result

    async def _expand_and_merge_async(self, query: str, main_result: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """异步版查询扩展合并"""
        intent = self._classify_fine_intent(query)
        variants = self.query_expander.expand(query, intent=intent)

        extra_variants = [v for v in variants if v != query]
        if not extra_variants:
            return main_result

        logger.info(f"Async query expansion: {len(extra_variants)} variants for '{query}'")

        existing_entities = set()
        for r in main_result.get("results", []):
            name = r.get("entity") or r.get("name", "")
            if name:
                existing_entities.add(name.lower())

        merged_results = list(main_result.get("results", []))
        for variant in extra_variants[:3]:
            try:
                if strategy == "global":
                    variant_result = self.global_search(variant)
                elif strategy == "local":
                    variant_result = await self.local_search_async(variant)
                else:
                    variant_result = await self.hybrid_search_async(variant)

                for r in variant_result.get("results", []):
                    name = r.get("entity") or r.get("name", "")
                    if name and name.lower() not in existing_entities:
                        existing_entities.add(name.lower())
                        r["_expanded_from"] = variant
                        merged_results.append(r)
            except Exception as e:
                logger.warning(f"Async query expansion variant '{variant}' failed: {e}")

        main_result["results"] = merged_results
        main_result["expanded_queries"] = extra_variants[:3]
        return main_result

    def explain_strategy(self, query: str) -> Dict[str, Any]:
        """解释为什么选择特定的检索策略"""
        intent = self._classify_query_intent(query)
        alpha = self._compute_dynamic_alpha(query)
        
        explanations = {
            "global": "查询包含全局指示词，适合使用社区摘要进行全局搜索",
            "local": "查询包含局部指示词，适合从实体出发进行精细化搜索",
            "hybrid": "查询意图不明确，使用混合检索策略",
        }
        
        return {
            "query": query,
            "detected_intent": intent,
            "alpha_weight": alpha,
            "strategy": intent,
            "explanation": explanations[intent],
        }


def drift_search(query: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    """便捷函数：执行DRIFT搜索"""
    searcher = DRIFTSearch()
    return searcher.search(query, strategy)


def explain_drift_strategy(query: str) -> Dict[str, Any]:
    """便捷函数：解释检索策略"""
    searcher = DRIFTSearch()
    return searcher.explain_strategy(query)