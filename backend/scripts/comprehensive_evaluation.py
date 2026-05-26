"""Comprehensive offline evaluation script for the medical knowledge graph system.

This script performs:
1. Functional testing
2. Performance testing
3. Compatibility testing
4. Problem diagnosis
5. Optimization recommendations
"""

import sys
import time
import json
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

import pytest

backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


@dataclass
class TestResult:
    test_name: str
    category: str
    status: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    warnings: int
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class OfflineEvaluator:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def add_result(self, result: TestResult):
        self.results.append(result)

    def run_functional_tests(self) -> Dict[str, Any]:
        """Run functional tests for all API endpoints."""
        print("\n" + "=" * 60)
        print("📋 FUNCTIONAL TESTS")
        print("=" * 60)

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        duration = (time.time() - start) * 1000

        passed = result.returncode == 0
        output = result.stdout + result.stderr

        passed_count = output.count(" PASSED")
        failed_count = output.count(" FAILED")

        test_result = TestResult(
            test_name="API Functional Tests",
            category="functional",
            status="passed" if passed else "failed",
            duration_ms=duration,
            details={
                "passed": passed_count,
                "failed": failed_count,
                "total": passed_count + failed_count,
            }
        )
        self.add_result(test_result)

        print(f"  ✅ Passed: {passed_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  ⏱️  Duration: {duration:.0f}ms")

        return {"passed": passed, "passed_count": passed_count, "failed_count": failed_count}

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE TESTS")
        print("=" * 60)

        results = {}

        # Test 1: Import time
        print("\n  Testing import times...")
        import_times = {}
        modules = [
            "src.api.routes",
            "src.workflow.graph",
            "src.retrieval.drift_search",
            "src.ingestion.kg_builder",
        ]

        for module in modules:
            start = time.time()
            try:
                __import__(module)
                duration = (time.time() - start) * 1000
                import_times[module] = duration
                status = "✅" if duration < 1000 else "⚠️"
                print(f"    {status} {module}: {duration:.0f}ms")
            except Exception as e:
                import_times[module] = -1
                print(f"    ❌ {module}: Error - {str(e)[:50]}")

        results["import_times"] = import_times

        # Test 2: Cache performance
        print("\n  Testing cache performance...")
        try:
            from src.core.cache import get_query_cache, cached

            cache = get_query_cache()
            cache_stats = cache.stats()
            results["cache_stats"] = cache_stats
            print(f"    Cache hits: {cache_stats.get('hits', 0)}")
            print(f"    Cache misses: {cache_stats.get('misses', 0)}")
            print(f"    Cache size: {cache_stats.get('size', 0)}")
        except Exception as e:
            print(f"    ⚠️ Cache test failed: {e}")
            results["cache_error"] = str(e)

        # Test 3: Memory usage estimation
        print("\n  Estimating memory usage...")
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            results["memory_mb"] = memory_mb
            status = "✅" if memory_mb < 500 else "⚠️"
            print(f"    {status} Current memory: {memory_mb:.1f}MB")
        except ImportError:
            print("    ⚠️ psutil not installed, skipping memory test")

        test_result = TestResult(
            test_name="Performance Benchmarks",
            category="performance",
            status="passed",
            duration_ms=sum(v for v in import_times.values() if isinstance(v, (int, float)) and v > 0),
            details=results
        )
        self.add_result(test_result)

        return results

    def run_compatibility_tests(self) -> Dict[str, Any]:
        """Check dependency compatibility."""
        print("\n" + "=" * 60)
        print("🔗 COMPATIBILITY TESTS")
        print("=" * 60)

        results = {"dependencies": {}, "warnings": []}

        # Check Python version
        python_version = sys.version_info
        print(f"\n  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        if python_version < (3, 9):
            results["warnings"].append("Python 3.9+ recommended")
        elif python_version >= (3, 12):
            results["warnings"].append("Python 3.12+ may have compatibility issues with some packages")

        # Check critical dependencies
        critical_deps = [
            ("fastapi", "0.100.0"),
            ("pydantic", "2.0.0"),
            ("neo4j", "5.0.0"),
            ("langchain", "0.1.0"),
            ("loguru", "0.7.0"),
        ]

        print("\n  Checking critical dependencies:")
        for pkg, min_version in critical_deps:
            try:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                print(f"    ✅ {pkg}: {version}")
                results["dependencies"][pkg] = {"version": version, "status": "ok"}
            except ImportError:
                print(f"    ❌ {pkg}: Not installed")
                results["dependencies"][pkg] = {"version": None, "status": "missing"}

        # Check optional dependencies
        optional_deps = [
            ("sentence_transformers", "For embedding generation"),
            ("spacy", "For NLP processing"),
            ("pytest", "For testing"),
            ("psutil", "For memory monitoring"),
        ]

        print("\n  Checking optional dependencies:")
        for pkg, purpose in optional_deps:
            try:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "installed")
                print(f"    ✅ {pkg}: {version} ({purpose})")
            except ImportError:
                print(f"    ⚠️ {pkg}: Not installed ({purpose})")

        test_result = TestResult(
            test_name="Compatibility Check",
            category="compatibility",
            status="passed",
            duration_ms=0,
            details=results
        )
        self.add_result(test_result)

        return results

    def diagnose_issues(self) -> Dict[str, Any]:
        """Diagnose potential issues and bottlenecks."""
        print("\n" + "=" * 60)
        print("🔍 PROBLEM DIAGNOSIS")
        print("=" * 60)

        issues = []
        warnings = []

        # Check 1: Database connection
        print("\n  Checking database connectivity...")
        try:
            from src.core.neo4j_client import Neo4jClient
            client = Neo4jClient()
            client.verify_connectivity()
            print("    ✅ Neo4j connection: OK")
        except Exception as e:
            issues.append(f"Neo4j connection failed: {str(e)[:100]}")
            print(f"    ❌ Neo4j connection: {str(e)[:100]}")

        # Check 2: Environment variables
        print("\n  Checking environment configuration...")
        import os
        required_env = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
        for env in required_env:
            if os.getenv(env):
                print(f"    ✅ {env}: Set")
            else:
                warnings.append(f"Environment variable {env} not set")
                print(f"    ⚠️ {env}: Not set")

        # Check 3: File permissions
        print("\n  Checking file permissions...")
        test_dirs = ["logs", "data", "uploads"]
        for d in test_dirs:
            path = Path(d)
            if path.exists():
                if os.access(path, os.W_OK):
                    print(f"    ✅ {d}/: Writable")
                else:
                    warnings.append(f"Directory {d}/ not writable")
                    print(f"    ⚠️ {d}/: Not writable")
            else:
                print(f"    ℹ️ {d}/: Does not exist")

        # Check 4: Circuit breaker status
        print("\n  Checking circuit breaker...")
        try:
            from src.core.circuit_breaker import get_circuit_breaker
            cb = get_circuit_breaker("neo4j")
            state = cb.state if hasattr(cb, 'state') else "unknown"
            print(f"    Circuit breaker state: {state}")
        except Exception as e:
            print(f"    ⚠️ Circuit breaker check failed: {e}")

        # Check 5: Error collector
        print("\n  Checking error collector...")
        try:
            from src.core.error_collector import get_error_collector
            collector = get_error_collector()
            stats = collector.get_stats()
            error_count = stats.get("total_errors", 0)
            if error_count > 0:
                warnings.append(f"Error collector has {error_count} recorded errors")
            print(f"    Total errors recorded: {error_count}")
        except Exception as e:
            print(f"    ⚠️ Error collector check failed: {e}")

        test_result = TestResult(
            test_name="Problem Diagnosis",
            category="diagnosis",
            status="passed" if not issues else "failed",
            duration_ms=0,
            details={
                "issues": issues,
                "warnings": warnings,
            }
        )
        self.add_result(test_result)

        return {"issues": issues, "warnings": warnings}

    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results."""
        recommendations = []

        for result in self.results:
            if result.category == "functional" and result.status == "failed":
                recommendations.append("🔴 Critical: Fix failing functional tests before deployment")

            if result.category == "performance":
                import_times = result.details.get("import_times", {})
                slow_imports = [m for m, t in import_times.items() if t > 1000]
                if slow_imports:
                    recommendations.append(f"⚠️ Optimize import time for: {', '.join(slow_imports)}")

            if result.category == "compatibility":
                missing = [k for k, v in result.details.get("dependencies", {}).items()
                          if v.get("status") == "missing"]
                if missing:
                    recommendations.append(f"📦 Install missing dependencies: {', '.join(missing)}")

            if result.category == "diagnosis":
                issues = result.details.get("issues", [])
                for issue in issues:
                    recommendations.append(f"🔧 Fix: {issue}")

        if not recommendations:
            recommendations.append("✅ All systems operating normally. No critical issues found.")

        return recommendations

    def generate_report(self) -> EvaluationReport:
        """Generate the final evaluation report."""
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        warnings = sum(len(r.details.get("warnings", [])) for r in self.results)

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            warnings=warnings,
            results=self.results,
            summary={
                "total_duration_s": time.time() - self.start_time,
                "pass_rate": f"{(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A",
            },
            recommendations=self.generate_recommendations()
        )

        return report

    def save_report(self, report: EvaluationReport, output_path: Optional[str] = None):
        """Save the report to a JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_results/offline_eval_{timestamp}.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report_dict = asdict(report)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Report saved to: {output_file}")
        return output_file

    def print_summary(self, report: EvaluationReport):
        """Print a summary of the evaluation."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\n  Total Tests: {report.total_tests}")
        print(f"  ✅ Passed: {report.passed}")
        print(f"  ❌ Failed: {report.failed}")
        print(f"  ⚠️  Warnings: {report.warnings}")
        print(f"  ⏱️  Duration: {report.summary['total_duration_s']:.1f}s")
        print(f"  📈 Pass Rate: {report.summary['pass_rate']}")

        print("\n" + "-" * 60)
        print("📋 RECOMMENDATIONS")
        print("-" * 60)
        for rec in report.recommendations:
            print(f"  {rec}")

        print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("🏥 MEDICAL KNOWLEDGE GRAPH - OFFLINE EVALUATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    evaluator = OfflineEvaluator()

    # Run all tests
    evaluator.run_functional_tests()
    evaluator.run_performance_tests()
    evaluator.run_compatibility_tests()
    evaluator.diagnose_issues()

    # Generate and save report
    report = evaluator.generate_report()
    evaluator.save_report(report)
    evaluator.print_summary(report)

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
