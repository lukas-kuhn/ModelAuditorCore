from setuptools import setup, find_packages

setup(
    name="model_auditor",
    version="0.1.0",
    description="A library for auditing ML models under distribution shifts",
    author="Lukas Kuhn",
    author_email="lukas.kuhn@dkfz-heidelberg.de",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "onnxruntime",
        "numpy",
        "scikit-learn",
        "rich",
        "MetricsReloaded @ git+https://github.com/csudre/MetricsReloaded.git",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 