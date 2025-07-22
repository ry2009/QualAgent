#!/usr/bin/env python3
"""
QualGent Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="qualgent",
    version="1.0.0",
    author="QualGent Team",
    author_email="team@qualgent.dev",
    description="Multi-Agent QA System for Mobile App Testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/qualgent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qualgent=main:main",
            "qualgent-demo=run_demo:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/qualgent/issues",
        "Source": "https://github.com/your-org/qualgent",
        "Documentation": "https://qualgent.readthedocs.io/",
    },
    keywords="qa testing mobile android automation ai agents llm",
    include_package_data=True,
    package_data={
        "qualgent": [
            "config/*.json",
            "tasks/*.json",
        ],
    },
) 