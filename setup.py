"""
Quick setup script for RLTrade

This script:
1. Checks dependencies
2. Initializes database
3. Verifies environment
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10+"""
    print("Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_env_file():
    """Check .env file exists"""
    print("\nChecking .env file...")
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Copy .env.example to .env and configure it")
        return False
    print("✅ .env file found")
    return True


def init_database():
    """Initialize database schema"""
    print("\nInitializing database...")
    try:
        # Import here to avoid issues if deps not installed
        sys.path.insert(0, str(Path("bot/src")))
        from data import init_db
        
        init_db()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = [
        "bot/models",
        "bot/logs",
        "bot/data",
        "logs/tensorboard"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def main():
    """Run setup"""
    print("="*60)
    print("RLTrade Setup")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        (".env File", check_env_file),
        ("Directories", create_directories),
        ("Database", init_database)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {str(e)}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    if all(result for _, result in results):
        print("\n🎉 Setup complete! You're ready to train.")
        print("\nNext steps:")
        print("  1. Activate virtual environment: .\\bot\\venv\\Scripts\\activate")
        print("  2. Test environment: python bot/main.py test-env")
        print("  3. Start training: python bot/main.py train --episodes 1000")
        return 0
    else:
        print("\n❌ Setup incomplete. Please fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
