"""
Load test for /analyze endpoint.
Run with: uv run python tests/load_test.py
"""
import asyncio
import httpx
import time


async def analyze_stock(client: httpx.AsyncClient, stock: str, request_id: int, port: int = 8002):
    """Make single analyze request"""
    start = time.time()
    try:
        response = await client.post(
            f"http://localhost:{port}/analyze",
            json={"stock": stock}
        )
        elapsed = time.time() - start
        return {
            "request_id": request_id,
            "status": response.status_code,
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }


async def run_load_test(num_requests: int = 20):
    """Run concurrent requests"""
    print(f"Starting load test with {num_requests} concurrent requests...")
    print("-" * 50)

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            analyze_stock(client, "AAPL", i)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    # Analyze results
    successes = [r for r in results if r["status"] == 200]
    capacity_errors = [r for r in results if r["status"] == 503]
    rate_limit_errors = [r for r in results if r["status"] == 429]
    other_errors = [r for r in results if r["status"] not in [200, 503, 429, "error"]]
    connection_errors = [r for r in results if r["status"] == "error"]

    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful (200): {len(successes)}")
    print(f"Capacity exceeded (503): {len(capacity_errors)}")
    print(f"Rate limited (429): {len(rate_limit_errors)}")
    print(f"Other errors: {len(other_errors)}")
    print(f"Connection errors: {len(connection_errors)}")

    if successes:
        avg_time = sum(r["elapsed"] for r in successes) / len(successes)
        print(f"\nAvg response time (successes): {avg_time:.2f}s")

    if capacity_errors:
        print(f"\n503 responses indicate concurrency limit is working!")

    # Print detailed results
    print(f"\n=== Detailed Results ===")
    for r in sorted(results, key=lambda x: x["request_id"]):
        status = r["status"]
        elapsed = r.get("elapsed", 0)
        error = r.get("error", "")
        if error:
            print(f"  Request {r['request_id']}: {status} - {error}")
        else:
            print(f"  Request {r['request_id']}: {status} ({elapsed:.2f}s)")


if __name__ == "__main__":
    asyncio.run(run_load_test(20))
