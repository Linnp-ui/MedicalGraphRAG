#!/usr/bin/env python3
"""
负载测试和故障注入测试套件
GraphRAG API 服务测试方案
"""

import asyncio
import aiohttp
import time
import json
import statistics
import random
import threading
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import subprocess
import signal

BASE_URL = "http://localhost:8000/api/v1"

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class LoadTestConfig:
    name: str
    concurrent_users: int
    requests_per_user: int
    ramp_up_seconds: float = 1.0
    think_time_ms: float = 100

@dataclass
class SuccessCriteria:
    max_response_time_ms: float = 500.0
    max_error_rate: float = 0.001
    min_throughput_rps: float = 10.0
    max_p95_response_time_ms: float = 1000.0
    max_p99_response_time_ms: float = 2000.0

class MetricsCollector:
    def __init__(self):
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        self.start_time: float = 0
        self.end_time: float = 0
    
    def add_result(self, result: TestResult):
        with self.lock:
            self.results.append(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            if not self.results:
                return {}
            
            response_times = [r.response_time_ms for r in self.results]
            successful = [r for r in self.results if r.success]
            failed = [r for r in self.results if not r.success]
            
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            
            return {
                "total_requests": len(self.results),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "error_rate": len(failed) / len(self.results) if self.results else 0,
                "response_time": {
                    "min": min(response_times) if response_times else 0,
                    "max": max(response_times) if response_times else 0,
                    "mean": statistics.mean(response_times) if response_times else 0,
                    "median": statistics.median(response_times) if response_times else 0,
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    "p50": sorted_times[int(n * 0.50)] if n > 0 else 0,
                    "p90": sorted_times[int(n * 0.90)] if n > 0 else 0,
                    "p95": sorted_times[int(n * 0.95)] if n > 0 else 0,
                    "p99": sorted_times[int(n * 0.99)] if n > 0 else 0,
                },
                "throughput_rps": len(self.results) / (self.end_time - self.start_time) if self.end_time > self.start_time else 0,
                "duration_seconds": self.end_time - self.start_time if self.end_time > self.start_time else 0,
            }
    
    def get_status_code_distribution(self) -> Dict[int, int]:
        with self.lock:
            distribution = {}
            for r in self.results:
                distribution[r.status_code] = distribution.get(r.status_code, 0) + 1
            return distribution
    
    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            endpoint_results = {}
            for r in self.results:
                if r.endpoint not in endpoint_results:
                    endpoint_results[r.endpoint] = []
                endpoint_results[r.endpoint].append(r)
            
            stats = {}
            for endpoint, results in endpoint_results.items():
                times = [r.response_time_ms for r in results]
                successful = [r for r in results if r.success]
                stats[endpoint] = {
                    "count": len(results),
                    "success_count": len(successful),
                    "error_rate": 1 - (len(successful) / len(results)) if results else 0,
                    "avg_response_time_ms": statistics.mean(times) if times else 0,
                    "max_response_time_ms": max(times) if times else 0,
                }
            return stats

class APIClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def request(self, method: str, endpoint: str, **kwargs) -> tuple:
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.request(method, url, **kwargs) as response:
                    response_time = (time.perf_counter() - start_time) * 1000
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
                    return response.status, response_time, data
        except asyncio.TimeoutError:
            response_time = (time.perf_counter() - start_time) * 1000
            return 408, response_time, {"error": "Request timeout"}
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return 0, response_time, {"error": str(e)}
    
    async def health_check(self) -> tuple:
        return await self.request("GET", "/health")
    
    async def get_graph_data(self, limit: int = 100) -> tuple:
        return await self.request("GET", f"/graph/data?limit={limit}")
    
    async def search_nodes(self, query: str, limit: int = 10) -> tuple:
        return await self.request("GET", f"/graph/search?query={query}&limit={limit}")
    
    async def get_node_detail(self, node_id: str) -> tuple:
        return await self.request("GET", f"/graph/node/{node_id}")
    
    async def get_node_neighbors(self, node_id: str, depth: int = 1) -> tuple:
        return await self.request("GET", f"/graph/node/{node_id}/neighbors?depth={depth}")
    
    async def query(self, question: str) -> tuple:
        return await self.request("POST", "/query", json={"question": question})
    
    async def get_schema(self) -> tuple:
        return await self.request("GET", "/schema")
    
    async def get_metrics(self) -> tuple:
        return await self.request("GET", "/metrics")

class LoadTestRunner:
    def __init__(self, config: LoadTestConfig, metrics: MetricsCollector):
        self.config = config
        self.metrics = metrics
        self.client = APIClient()
    
    async def user_scenario(self, user_id: int) -> List[TestResult]:
        results = []
        
        for i in range(self.config.requests_per_user):
            scenario = random.choice([
                self.scenario_browse_graph,
                self.scenario_search,
                self.scenario_query,
            ])
            
            try:
                result = await scenario(user_id, i)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                results.append(TestResult(
                    endpoint="unknown",
                    method="GET",
                    status_code=0,
                    response_time_ms=0,
                    success=False,
                    error=str(e)
                ))
            
            if self.config.think_time_ms > 0:
                await asyncio.sleep(self.config.think_time_ms / 1000)
        
        return results
    
    async def scenario_browse_graph(self, user_id: int, request_id: int) -> TestResult:
        status, response_time, data = await self.client.get_graph_data(limit=50)
        return TestResult(
            endpoint="/graph/data",
            method="GET",
            status_code=status,
            response_time_ms=response_time,
            success=200 <= status < 300
        )
    
    async def scenario_search(self, user_id: int, request_id: int) -> List[TestResult]:
        results = []
        
        search_terms = ["Person", "Document", "Organization", "test", "data"]
        query = random.choice(search_terms)
        
        status, response_time, data = await self.client.search_nodes(query)
        results.append(TestResult(
            endpoint="/graph/search",
            method="GET",
            status_code=status,
            response_time_ms=response_time,
            success=200 <= status < 300
        ))
        
        if 200 <= status < 300 and isinstance(data, dict) and "results" in data:
            node_results = data.get("results", [])
            if node_results:
                node_id = node_results[0].get("id")
                if node_id:
                    status2, response_time2, _ = await self.client.get_node_detail(str(node_id))
                    results.append(TestResult(
                        endpoint="/graph/node/{id}",
                        method="GET",
                        status_code=status2,
                        response_time_ms=response_time2,
                        success=200 <= status2 < 300
                    ))
        
        return results
    
    async def scenario_query(self, user_id: int, request_id: int) -> TestResult:
        questions = [
            "What entities are in the graph?",
            "Show me related documents",
            "Find connections between nodes",
        ]
        question = random.choice(questions)
        
        status, response_time, data = await self.client.query(question)
        return TestResult(
            endpoint="/query",
            method="POST",
            status_code=status,
            response_time_ms=response_time,
            success=200 <= status < 300
        )
    
    async def run(self) -> Dict[str, Any]:
        self.metrics.start_time = time.time()
        
        tasks = []
        for i in range(self.config.concurrent_users):
            delay = (i / self.config.concurrent_users) * self.config.ramp_up_seconds
            task = asyncio.create_task(self.delayed_user_scenario(i, delay))
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
        
        for user_results in all_results:
            for result in user_results:
                self.metrics.add_result(result)
        
        self.metrics.end_time = time.time()
        return self.metrics.get_statistics()
    
    async def delayed_user_scenario(self, user_id: int, delay: float):
        await asyncio.sleep(delay)
        return await self.user_scenario(user_id)

class FaultInjector:
    def __init__(self):
        self.active_faults: List[str] = []
    
    async def inject_network_latency(self, latency_ms: int = 500) -> bool:
        print(f"[故障注入] 模拟网络延迟: {latency_ms}ms")
        await asyncio.sleep(latency_ms / 1000)
        return True
    
    async def inject_timeout(self, client: APIClient, timeout_seconds: float = 1.0) -> Dict[str, Any]:
        print(f"[故障注入] 模拟超时: {timeout_seconds}s")
        original_timeout = client.timeout
        client.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        
        try:
            status, response_time, data = await client.health_check()
            return {
                "fault_type": "timeout",
                "timeout_seconds": timeout_seconds,
                "status_code": status,
                "response_time_ms": response_time,
                "success": status == 200
            }
        finally:
            client.timeout = original_timeout
    
    async def inject_high_load(self, duration_seconds: int = 10, rps: int = 100) -> Dict[str, Any]:
        print(f"[故障注入] 高负载测试: {rps} RPS 持续 {duration_seconds}s")
        
        metrics = MetricsCollector()
        client = APIClient()
        
        async def rapid_request():
            status, response_time, _ = await client.health_check()
            return TestResult(
                endpoint="/health",
                method="GET",
                status_code=status,
                response_time_ms=response_time,
                success=200 <= status < 300
            )
        
        metrics.start_time = time.time()
        
        tasks = []
        for _ in range(rps * duration_seconds):
            tasks.append(rapid_request())
            await asyncio.sleep(1.0 / rps)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, TestResult):
                metrics.add_result(result)
        
        metrics.end_time = time.time()
        
        return {
            "fault_type": "high_load",
            "duration_seconds": duration_seconds,
            "target_rps": rps,
            "statistics": metrics.get_statistics()
        }
    
    async def inject_connection_failure(self, client: APIClient) -> Dict[str, Any]:
        print("[故障注入] 模拟连接失败")
        
        original_url = client.base_url
        client.base_url = "http://localhost:9999"
        
        try:
            status, response_time, data = await client.health_check()
            return {
                "fault_type": "connection_failure",
                "status_code": status,
                "response_time_ms": response_time,
                "success": False,
                "error": data.get("error") if isinstance(data, dict) else str(data)
            }
        finally:
            client.base_url = original_url

class TestReportGenerator:
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        load_test_results: Dict[str, Dict[str, Any]],
        fault_test_results: List[Dict[str, Any]],
        criteria: SuccessCriteria
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"test_report_{timestamp}.md")
        
        report = self._build_report(load_test_results, fault_test_results, criteria)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        json_path = os.path.join(self.output_dir, f"test_results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "load_tests": load_test_results,
                "fault_tests": fault_test_results,
                "criteria": {
                    "max_response_time_ms": criteria.max_response_time_ms,
                    "max_error_rate": criteria.max_error_rate,
                    "min_throughput_rps": criteria.min_throughput_rps,
                }
            }, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def _build_report(
        self,
        load_test_results: Dict[str, Dict[str, Any]],
        fault_test_results: List[Dict[str, Any]],
        criteria: SuccessCriteria
    ) -> str:
        lines = [
            "# GraphRAG API 负载与故障注入测试报告",
            f"\n**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. 测试环境",
            "",
            "| 配置项 | 值 |",
            "|--------|-----|",
            f"| 服务地址 | {BASE_URL} |",
            "| 服务框架 | FastAPI + Uvicorn |",
            "| 数据库 | Neo4j (bolt://localhost:7687) |",
            "| 测试工具 | Python asyncio + aiohttp |",
            "",
            "---",
            "",
            "## 2. 成功标准",
            "",
            "| 指标 | 阈值 |",
            "|------|------|",
            f"| 平均响应时间 | < {criteria.max_response_time_ms}ms |",
            f"| 错误率 | < {criteria.max_error_rate * 100}% |",
            f"| 最小吞吐量 | > {criteria.min_throughput_rps} RPS |",
            f"| P95响应时间 | < {criteria.max_p95_response_time_ms}ms |",
            f"| P99响应时间 | < {criteria.max_p99_response_time_ms}ms |",
            "",
            "---",
            "",
            "## 3. 负载测试结果",
            "",
        ]
        
        for test_name, stats in load_test_results.items():
            lines.extend([
                f"### 3.{list(load_test_results.keys()).index(test_name) + 1} {test_name}",
                "",
                "| 指标 | 值 | 状态 |",
                "|------|-----|------|",
            ])
            
            response_stats = stats.get("response_time", {})
            error_rate = stats.get("error_rate", 0)
            throughput = stats.get("throughput_rps", 0)
            
            avg_time = response_stats.get("mean", 0)
            p95_time = response_stats.get("p95", 0)
            p99_time = response_stats.get("p99", 0)
            
            avg_status = "✅" if avg_time < criteria.max_response_time_ms else "❌"
            error_status = "✅" if error_rate < criteria.max_error_rate else "❌"
            throughput_status = "✅" if throughput >= criteria.min_throughput_rps else "❌"
            p95_status = "✅" if p95_time < criteria.max_p95_response_time_ms else "❌"
            p99_status = "✅" if p99_time < criteria.max_p99_response_time_ms else "❌"
            
            lines.extend([
                f"| 总请求数 | {stats.get('total_requests', 0)} | - |",
                f"| 成功请求 | {stats.get('successful_requests', 0)} | - |",
                f"| 失败请求 | {stats.get('failed_requests', 0)} | - |",
                f"| 错误率 | {error_rate * 100:.2f}% | {error_status} |",
                f"| 吞吐量 | {throughput:.2f} RPS | {throughput_status} |",
                f"| 平均响应时间 | {avg_time:.2f}ms | {avg_status} |",
                f"| 最小响应时间 | {response_stats.get('min', 0):.2f}ms | - |",
                f"| 最大响应时间 | {response_stats.get('max', 0):.2f}ms | - |",
                f"| P50响应时间 | {response_stats.get('p50', 0):.2f}ms | - |",
                f"| P90响应时间 | {response_stats.get('p90', 0):.2f}ms | - |",
                f"| P95响应时间 | {p95_time:.2f}ms | {p95_status} |",
                f"| P99响应时间 | {p99_time:.2f}ms | {p99_status} |",
                f"| 标准差 | {response_stats.get('std_dev', 0):.2f}ms | - |",
                f"| 测试时长 | {stats.get('duration_seconds', 0):.2f}s | - |",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "## 4. 故障注入测试结果",
            "",
        ])
        
        for i, fault in enumerate(fault_test_results, 1):
            fault_type = fault.get("fault_type", "unknown")
            lines.extend([
                f"### 4.{i} {fault_type}",
                "",
                "```json",
                json.dumps(fault, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "## 5. 测试总结",
            "",
            self._generate_summary(load_test_results, fault_test_results, criteria),
            "",
            "---",
            "",
            "## 6. 优化建议",
            "",
        ])
        
        recommendations = self._generate_recommendations(load_test_results, fault_test_results, criteria)
        for rec in recommendations:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def _generate_summary(
        self,
        load_test_results: Dict[str, Dict[str, Any]],
        fault_test_results: List[Dict[str, Any]],
        criteria: SuccessCriteria
    ) -> str:
        all_passed = True
        issues = []
        
        for test_name, stats in load_test_results.items():
            response_stats = stats.get("response_time", {})
            error_rate = stats.get("error_rate", 0)
            throughput = stats.get("throughput_rps", 0)
            avg_time = response_stats.get("mean", 0)
            p95_time = response_stats.get("p95", 0)
            
            if avg_time >= criteria.max_response_time_ms:
                all_passed = False
                issues.append(f"{test_name}: 平均响应时间 {avg_time:.2f}ms 超过阈值")
            if error_rate >= criteria.max_error_rate:
                all_passed = False
                issues.append(f"{test_name}: 错误率 {error_rate * 100:.2f}% 超过阈值")
            if throughput < criteria.min_throughput_rps:
                all_passed = False
                issues.append(f"{test_name}: 吞吐量 {throughput:.2f} RPS 低于阈值")
            if p95_time >= criteria.max_p95_response_time_ms:
                all_passed = False
                issues.append(f"{test_name}: P95响应时间 {p95_time:.2f}ms 超过阈值")
        
        if all_passed:
            return "✅ **所有负载测试通过**，服务性能满足预期要求。"
        else:
            return "❌ **部分测试未通过**，发现以下问题：\n\n" + "\n".join(f"- {issue}" for issue in issues)
    
    def _generate_recommendations(
        self,
        load_test_results: Dict[str, Dict[str, Any]],
        fault_test_results: List[Dict[str, Any]],
        criteria: SuccessCriteria
    ) -> List[str]:
        recommendations = []
        
        for test_name, stats in load_test_results.items():
            response_stats = stats.get("response_time", {})
            error_rate = stats.get("error_rate", 0)
            avg_time = response_stats.get("mean", 0)
            max_time = response_stats.get("max", 0)
            
            if avg_time > criteria.max_response_time_ms * 0.8:
                recommendations.append(f"考虑增加缓存策略以降低 {test_name} 的平均响应时间")
            
            if max_time > criteria.max_p99_response_time_ms:
                recommendations.append(f"调查 {test_name} 中响应时间异常高的请求，可能存在慢查询")
            
            if error_rate > criteria.max_error_rate * 0.5:
                recommendations.append(f"检查 {test_name} 中的错误日志，优化错误处理机制")
        
        if not recommendations:
            recommendations.append("服务性能表现良好，建议持续监控生产环境指标")
            recommendations.append("考虑实施自动化性能测试流水线")
            recommendations.append("定期进行容量规划评估")
        
        return recommendations

async def run_all_tests():
    print("=" * 60)
    print("GraphRAG API 负载与故障注入测试套件")
    print("=" * 60)
    
    criteria = SuccessCriteria(
        max_response_time_ms=500.0,
        max_error_rate=0.001,
        min_throughput_rps=10.0,
        max_p95_response_time_ms=1000.0,
        max_p99_response_time_ms=2000.0
    )
    
    load_configs = [
        LoadTestConfig(
            name="基准负载测试 (10用户)",
            concurrent_users=10,
            requests_per_user=5,
            ramp_up_seconds=2.0,
            think_time_ms=100
        ),
        LoadTestConfig(
            name="中等负载测试 (30用户)",
            concurrent_users=30,
            requests_per_user=10,
            ramp_up_seconds=5.0,
            think_time_ms=50
        ),
        LoadTestConfig(
            name="峰值负载测试 (50用户)",
            concurrent_users=50,
            requests_per_user=5,
            ramp_up_seconds=3.0,
            think_time_ms=20
        ),
    ]
    
    load_test_results = {}
    
    print("\n[阶段1] 服务健康检查")
    print("-" * 40)
    
    client = APIClient()
    status, response_time, data = await client.health_check()
    print(f"健康检查状态: {status}")
    print(f"响应时间: {response_time:.2f}ms")
    
    if status != 200:
        print("❌ 服务不可用，请先启动服务")
        return
    
    print("✅ 服务运行正常")
    
    print("\n[阶段2] 负载测试")
    print("-" * 40)
    
    for config in load_configs:
        print(f"\n执行: {config.name}")
        print(f"  - 并发用户: {config.concurrent_users}")
        print(f"  - 每用户请求数: {config.requests_per_user}")
        print(f"  - 爬升时间: {config.ramp_up_seconds}s")
        
        metrics = MetricsCollector()
        runner = LoadTestRunner(config, metrics)
        
        start_time = time.time()
        stats = await runner.run()
        duration = time.time() - start_time
        
        load_test_results[config.name] = stats
        
        print(f"  完成! 耗时: {duration:.2f}s")
        print(f"  总请求: {stats.get('total_requests', 0)}")
        print(f"  错误率: {stats.get('error_rate', 0) * 100:.2f}%")
        print(f"  吞吐量: {stats.get('throughput_rps', 0):.2f} RPS")
        print(f"  平均响应时间: {stats.get('response_time', {}).get('mean', 0):.2f}ms")
    
    print("\n[阶段3] 故障注入测试")
    print("-" * 40)
    
    fault_injector = FaultInjector()
    fault_test_results = []
    
    print("\n执行: 超时测试")
    timeout_result = await fault_injector.inject_timeout(client, timeout_seconds=1.0)
    fault_test_results.append(timeout_result)
    print(f"  结果: {timeout_result}")
    
    print("\n执行: 连接失败测试")
    connection_result = await fault_injector.inject_connection_failure(client)
    fault_test_results.append(connection_result)
    print(f"  结果: {connection_result}")
    
    print("\n执行: 快速高负载测试")
    high_load_result = await fault_injector.inject_high_load(duration_seconds=5, rps=50)
    fault_test_results.append(high_load_result)
    print(f"  总请求: {high_load_result.get('statistics', {}).get('total_requests', 0)}")
    
    print("\n[阶段4] 生成测试报告")
    print("-" * 40)
    
    report_generator = TestReportGenerator()
    report_path = report_generator.generate_report(load_test_results, fault_test_results, criteria)
    
    print(f"✅ 测试报告已生成: {report_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return load_test_results, fault_test_results, report_path

if __name__ == "__main__":
    asyncio.run(run_all_tests())
