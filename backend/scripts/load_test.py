"""Load test script for GraphRAG API"""

import asyncio
import aiohttp
import time
import random
from typing import List, Dict
import statistics


BASE_URL = "http://localhost:8000/api/v1"
CONCURRENT_USERS = 50
TEST_DURATION = 60  # seconds
RAMP_UP_TIME = 10  # seconds


MEDICAL_QUESTIONS = [
    "高血压的主要症状有哪些？",
    "糖尿病如何治疗？",
    "阿司匹林的副作用是什么？",
    "感冒发烧怎么办？",
    "高血压用什么药效果好？",
    "胸痛可能是什么原因？",
    "头晕是怎么回事？",
    "血常规检查包括哪些项目？",
    "CT检查需要注意什么？",
    "如何预防心脑血管疾病？",
    "失眠怎么治疗？",
    "胃痛吃什么药？",
    "痛风的饮食禁忌有哪些？",
    "甲状腺功能检查指标有哪些？",
    "冠心病的早期症状？",
]


async def make_request(session: aiohttp.ClientSession, question: str) -> Dict:
    """Make a single query request"""
    start = time.perf_counter()
    try:
        async with session.post(
            f"{BASE_URL}/query",
            json={"question": question, "mode": "hybrid"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            await response.text()
            duration = (time.perf_counter() - start) * 1000
            return {
                "status": response.status,
                "duration_ms": duration,
                "success": response.status == 200,
            }
    except asyncio.TimeoutError:
        duration = (time.perf_counter() - start) * 1000
        return {"status": 0, "duration_ms": duration, "success": False, "error": "timeout"}
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return {"status": 0, "duration_ms": duration, "success": False, "error": str(e)}


async def user_session(session: aiohttp.ClientSession, user_id: int, results: List[Dict]):
    """Simulate a single user making requests"""
    while True:
        question = random.choice(MEDICAL_QUESTIONS)
        result = await make_request(session, question)
        result["user_id"] = user_id
        results.append(result)
        
        # Think time between requests
        await asyncio.sleep(random.uniform(0.5, 2.0))


async def run_warmup(session: aiohttp.ClientSession, count: int = 3):
    """Send a few warmup requests"""
    for _ in range(count):
        question = random.choice(MEDICAL_QUESTIONS)
        await make_request(session, question)


async def run_load_test():
    """Run the load test"""
    print(f"Starting load test: {CONCURRENT_USERS} users for {TEST_DURATION}s")
    print(f"Ramp-up: {RAMP_UP_TIME}s")
    print("-" * 50)
    
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warm up
        print("Warming up...")
        await run_warmup(session, 5)
        print("Warmup complete")
        
        # Start user sessions
        results = []
        tasks = []
        
        for i in range(CONCURRENT_USERS):
            task = asyncio.create_task(user_session(session, i, results))
            tasks.append(task)
            if i < CONCURRENT_USERS - 1:
                await asyncio.sleep(RAMP_UP_TIME / CONCURRENT_USERS)
        
        # Run for test duration
        await asyncio.sleep(TEST_DURATION)
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print results
        print("\n" + "=" * 50)
        print("LOAD TEST RESULTS")
        print("=" * 50)
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Error rate: {len(failed)/len(results)*100:.2f}%")
        
        if successful:
            durations = [r["duration_ms"] for r in successful]
            print(f"\nResponse times (successful):")
            print(f"  Min: {min(durations):.1f}ms")
            print(f"  Max: {max(durations):.1f}ms")
            print(f"  Mean: {statistics.mean(durations):.1f}ms")
            print(f"  Median: {statistics.median(durations):.1f}ms")
            print(f"  P50: {statistics.median(durations):.1f}ms")
            print(f"  P90: {sorted(durations)[int(len(durations)*0.9)]:.1f}ms")
            print(f"  P95: {sorted(durations)[int(len(durations)*0.95)]:.1f}ms")
            print(f"  P99: {sorted(durations)[int(len(durations)*0.99)]:.1f}ms")
        
        if failed:
            error_types = {}
            for r in failed:
                err = r.get("error", f"HTTP_{r['status']}")
                error_types[err] = error_types.get(err, 0) + 1
            print(f"\nError breakdown:")
            for err, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {err}: {count}")
        
        # Throughput
        total_time = TEST_DURATION
        print(f"\nThroughput: {len(results)/total_time:.2f} req/s")
        
        return results


if __name__ == "__main__":
    asyncio.run(run_load_test())