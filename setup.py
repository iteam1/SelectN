from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [ l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]

setup(
    name="selectn",
    version="0.1.0",
    author="iteam1",
    author_email="info@selectn.dev",
    description="AI-powered representative document sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iteam1/selectN",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "pylint>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "selectn=selectn.cli.cli:main",
        ],
    },
)