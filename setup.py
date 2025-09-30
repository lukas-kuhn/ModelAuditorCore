from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model_auditor",
    version="0.2.0",
    description="A library for auditing ML models under distribution shifts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lukas Kuhn",
    author_email="lukas.kuhn@dkfz-heidelberg.de",
    url="https://github.com/lukaskuhn/ModelAuditorCore",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "onnxruntime",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "rich",
        "MetricsReloaded @ git+https://github.com/csudre/MetricsReloaded.git",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 