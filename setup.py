from setuptools import setup, find_packages

setup(
    name="langgraph-benchmarks",
    version="0.1.0",
    description="Benchmarking framework for LangGraph multi-agent systems",
    author="LangGraph Bench Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "psutil",
        "numpy",
        "matplotlib",
        "pandas",
        "networkx",
        "langchain",
        "langchain-openai",
        "langgraph",
        "python-dotenv",
    ],
    extras_require={
        "gpu": ["gputil"],
        "dev": ["pytest", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)
