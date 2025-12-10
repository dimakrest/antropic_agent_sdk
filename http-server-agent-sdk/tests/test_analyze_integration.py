"""
Integration tests for the /analyze swing trading endpoint.

These tests require:
1. ANTHROPIC_API_KEY environment variable
2. Technical Analysis API running on localhost:8093
3. HTTP server running on localhost:8000

Run with: uv run python tests/test_analyze_integration.py
"""

import os
import sys
import httpx
import asyncio


async def test_analyze_valid_symbol():
    """Test analysis of a valid stock symbol"""
    print("\n=== Test: Analyze Valid Symbol (AAPL) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "AAPL"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["stock"] == "AAPL", "Stock ticker mismatch"
    assert data["recommendation"] in ["Buy", "Not Buy"], "Invalid recommendation"
    assert 0 <= data["confidence_score"] <= 100, "Confidence out of range"
    assert len(data["reasoning"]) > 0, "Missing reasoning"

    if data["recommendation"] == "Buy":
        assert data["entry_price"] > 0, "Buy recommendation needs entry price"
        assert data["stop_loss"] > 0, "Buy recommendation needs stop loss"
        assert data["take_profit"] > 0, "Buy recommendation needs take profit"
        assert data["stop_loss"] < data["entry_price"], "Stop loss should be below entry"
        assert data["take_profit"] > data["entry_price"], "Take profit should be above entry"

    print("✓ Test passed")
    return True


async def test_analyze_invalid_symbol():
    """Test analysis of an invalid stock symbol"""
    print("\n=== Test: Analyze Invalid Symbol (INVALIDXYZ) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "INVALIDXYZ"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["recommendation"] == "Not Buy", "Invalid symbol should be Not Buy"

    print("✓ Test passed")
    return True


async def test_analyze_lowercase_symbol():
    """Test that lowercase symbols work correctly"""
    print("\n=== Test: Analyze Lowercase Symbol (msft) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "msft"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Stock in response: {data['stock']}")

    assert response.status_code == 200
    # Response should have uppercase ticker
    assert data["stock"].upper() == "MSFT"

    print("✓ Test passed")
    return True


async def check_prerequisites():
    """Check that required services are running"""
    print("Checking prerequisites...")

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set")
        return False
    print("✓ ANTHROPIC_API_KEY is set")

    # Check Technical Analysis API (correct endpoint: /api/v1/stocks/analysis/{symbol})
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8093/api/v1/stocks/analysis/AAPL",
                timeout=5.0
            )
        if response.status_code == 200:
            print("✓ Technical Analysis API is running on localhost:8093")
        else:
            print(f"✗ Technical Analysis API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Technical Analysis API not reachable: {e}")
        return False

    # Check HTTP server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8000/health",
                timeout=5.0
            )
        if response.status_code == 200:
            print("✓ HTTP server is running on localhost:8000")
        else:
            print(f"✗ HTTP server health check returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ HTTP server not reachable: {e}")
        return False

    return True


async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Swing Trading Agent Integration Tests")
    print("=" * 60)

    if not await check_prerequisites():
        print("\n✗ Prerequisites not met. Exiting.")
        sys.exit(1)

    tests = [
        test_analyze_valid_symbol,
        test_analyze_invalid_symbol,
        test_analyze_lowercase_symbol,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
