#!/usr/bin/env python3
"""
Railway environment check script
"""
import os
import sys
import platform

def check_environment():
    """Check Railway environment variables and system info."""
    print("🔍 Railway Environment Check")
    print("=" * 40)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Environment variables
    print("\n📋 Environment Variables:")
    required_vars = ['DISCORD_TOKEN', 'OPENAI_API_KEY']
    optional_vars = ['RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID', 'PORT']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * min(len(value), 8)}...")
        else:
            print(f"  ❌ {var}: NOT SET")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ℹ️  {var}: {value}")
        else:
            print(f"  ⚠️  {var}: NOT SET")
    
    # Port configuration
    port = os.getenv('PORT', '8000')
    print(f"\n🌐 Port configuration: {port}")
    
    # File system check
    print("\n📁 File System Check:")
    current_dir = os.getcwd()
    print(f"  Current directory: {current_dir}")
    
    files = os.listdir('.')
    print(f"  Files in directory: {len(files)}")
    for file in files:
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"    📄 {file} ({size} bytes)")
        else:
            print(f"    📁 {file}/")
    
    print("\n✅ Environment check complete!")

if __name__ == "__main__":
    check_environment() 