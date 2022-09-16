from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="brokorli_re",
    version="1.0.0",
    description="Korean relation extraction package within a brokorli project to construct knowledge graph",
    author="chnaaam",
    url="https://github.com/brokorli",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8.0",
    zip_safe=False
)