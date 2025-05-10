from setuptools import find_packages, setup

setup(
    name="small_llm_reasoning",
    version="0.0.1",
    description="Enhancing reasoning of small llms.",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    python_requires=">=3.11.0",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "together",
        "openai",
        "ollama",
        "torch==2.4.0",
        "transformers",
        "vllm==0.6.0",
        "vllm-flash-attn==2.6.1",
        "datasets==3.2.0"
    ],
)