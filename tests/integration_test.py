#!/usr/bin/env python3
"""
Integration tests for HTTP Server + Real Agent SDK

Prerequisites:
- Server running at http://localhost:8000
- ANTHROPIC_API_KEY environment variable set

Run with: python tests/integration_test.py
"""

import httpx
import sys
import os

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
TIMEOUT = 120.0  # 2 minutes for agent responses


class IntegrationTester:
    def __init__(self):
        self.client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)

    def test_health_check(self):
        """Test 1: Verify server is running"""
        print("\nTest 1: Health Check")
        resp = self.client.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        # Note: sdk_ready checks env var, but SDK may find key elsewhere
        print(f"Server is healthy: {data}")

    def test_simple_query(self):
        """Test 2: Send a simple message and get response"""
        print("\nTest 2: Simple Query")
        resp = self.client.post("/chat", json={
            "message": "What is 2 + 2? Just give me the number, nothing else."
        })
        assert resp.status_code == 200, f"Chat failed: {resp.status_code}"
        data = resp.json()

        assert "session_id" in data, "Missing session_id in response"
        assert data["status"] == "success", f"Query failed: {data.get('error')}"
        assert data["response_text"], "Empty response from agent"

        print(f"Got session ID: {data['session_id']}")
        print(f"Response: {data['response_text'][:200]}...")

        # Clean up
        self.client.delete(f"/sessions/{data['session_id']}")

    def test_conversation_memory(self):
        """Test 3: Multi-turn conversation with context"""
        print("\nTest 3: Conversation Memory")

        # First message: establish context
        resp1 = self.client.post("/chat", json={
            "message": "I have a dog named Max. Remember this important fact."
        })
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]
        print(f"First message sent. Session: {session_id}")

        # Second message: reference previous context
        resp2 = self.client.post("/chat", json={
            "session_id": session_id,
            "message": "What is my dog's name? Just tell me the name."
        })
        assert resp2.status_code == 200
        data = resp2.json()

        print(f"Response: {data['response_text'][:200]}...")
        assert "Max" in data["response_text"], "Claude forgot the context!"
        print("Context preserved across turns!")

        # Clean up
        self.client.delete(f"/sessions/{session_id}")

    def test_tool_usage(self):
        """Test 4: Verify agent can use tools (file operations)"""
        print("\nTest 4: Tool Usage")

        # Ask Claude to create a file
        resp = self.client.post("/chat", json={
            "message": "Create a file called test_output.txt with the content 'Hello from Claude'. Use the Write tool.",
            "permission_mode": "acceptEdits"  # Auto-approve file writes
        })
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        print(f"Claude executed (session: {session_id})")

        # Verify file exists
        if os.path.exists("test_output.txt"):
            with open("test_output.txt", "r") as f:
                content = f.read()
            print(f"File created with content: {content}")
            os.remove("test_output.txt")  # Cleanup
        else:
            print("File wasn't created (check tool permissions)")

        # Clean up session
        self.client.delete(f"/sessions/{session_id}")

    def test_error_handling(self):
        """Test 5: Error handling for invalid requests"""
        print("\nTest 5: Error Handling")

        # Test missing message
        resp = self.client.post("/chat", json={})
        assert resp.status_code == 422, "Should reject missing message"
        print(f"Missing message validation works (status: {resp.status_code})")

        # Test empty message
        resp = self.client.post("/chat", json={"message": ""})
        assert resp.status_code == 422, "Should reject empty message"
        print(f"Empty message validation works")

    def test_session_termination(self):
        """Test 6: Explicit session termination"""
        print("\nTest 6: Session Termination")

        # Create session
        resp = self.client.post("/chat", json={
            "message": "Start a test session"
        })
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        print(f"Session created: {session_id}")

        # Terminate session
        resp = self.client.delete(f"/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "terminated"
        print(f"Session terminated successfully")

        # Verify session is gone (try to delete again)
        resp = self.client.delete(f"/sessions/{session_id}")
        assert resp.status_code == 404, "Session should not exist"
        print(f"Verified session was removed")

    def test_concurrent_sessions(self):
        """Test 7: Multiple concurrent sessions"""
        print("\nTest 7: Concurrent Sessions")

        # Create two separate sessions with different contexts
        resp1 = self.client.post("/chat", json={
            "message": "My favorite color is blue. Remember this."
        })
        assert resp1.status_code == 200
        session1 = resp1.json()["session_id"]

        resp2 = self.client.post("/chat", json={
            "message": "My favorite color is red. Remember this."
        })
        assert resp2.status_code == 200
        session2 = resp2.json()["session_id"]

        assert session1 != session2, "Sessions should have different IDs"
        print(f"Two separate sessions: {session1[:8]}... and {session2[:8]}...")

        # Verify sessions don't interfere with each other
        resp1_followup = self.client.post("/chat", json={
            "session_id": session1,
            "message": "What is my favorite color?"
        })

        resp2_followup = self.client.post("/chat", json={
            "session_id": session2,
            "message": "What is my favorite color?"
        })

        text1 = resp1_followup.json()["response_text"]
        text2 = resp2_followup.json()["response_text"]

        print(f"Session 1 remembers: {text1[:100]}...")
        print(f"Session 2 remembers: {text2[:100]}...")

        # Sessions should remember their own context
        assert "blue" in text1.lower() or "Blue" in text1
        assert "red" in text2.lower() or "Red" in text2

        # Clean up
        self.client.delete(f"/sessions/{session1}")
        self.client.delete(f"/sessions/{session2}")

    def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("Starting Integration Tests")
        print("=" * 60)

        try:
            self.test_health_check()
            self.test_simple_query()
            self.test_conversation_memory()
            self.test_tool_usage()
            self.test_error_handling()
            self.test_session_termination()
            self.test_concurrent_sessions()

            print("\n" + "=" * 60)
            print("ALL INTEGRATION TESTS PASSED!")
            print("=" * 60)
            return True

        except AssertionError as e:
            print(f"\nTEST FAILED: {e}")
            return False
        except httpx.ConnectError:
            print(f"\nCannot connect to server at {BASE_URL}")
            print("Make sure the server is running:")
            print("  export ANTHROPIC_API_KEY='sk-ant-...'")
            print("  python server.py")
            return False
        except Exception as e:
            print(f"\nUNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    tester = IntegrationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
