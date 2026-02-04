"""Setup script for RLTrade bot"""

from setuptools import setup, find_packages

setup(
    name="rltrade-bot",
    version="0.1.0",
    description="Reinforcement Learning Trading Bot for Polymarket",
    author="RLTrade Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "stable-baselines3>=2.2.1",
        "gymnasium>=0.29.1",
        "torch>=2.1.2",
        "numpy>=1.26.3",
        "pandas>=2.1.4",
        "sqlalchemy>=2.0.25",
        "psycopg2-binary>=2.9.9",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "click>=8.1.7",
        "rich>=13.7.0",
        "tensorboard>=2.15.1",
    ],
    entry_points={
        "console_scripts": [
            "rltrade=main:cli",
        ],
    },
)
