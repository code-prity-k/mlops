from setuptools import setup, find_packages

setup(
    name="mnist_classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "pillow",
    ],
    extras_require={
        "test": ["pytest"],
    },
    python_requires=">=3.8",
) 