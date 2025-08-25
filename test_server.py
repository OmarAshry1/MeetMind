#!/usr/bin/env python3
"""
Simple test script to verify Flask server functionality
"""
import requests
import time
import sys

def test_flask_server():
    """Test if Flask server is responding."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Flask server...")
    
    # Wait a bit for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
            
        # Test detailed health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Detailed health endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Detailed health endpoint failed with status {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_flask_server()
    sys.exit(0 if success else 1) 