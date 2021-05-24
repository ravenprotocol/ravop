from setuptools import setup, find_packages

setup(
    name='ravop',
    version='0.1-alpha',
    packages=find_packages(),
    install_requires=[
        "numpy==1.20.1"
    ],
    dependency_links=[
        "https://github.com/ravenprotocol/ravcom.git@0.1-alpha"
    ]
)
