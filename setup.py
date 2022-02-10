from setuptools import setup, find_packages

setup(
    name='ravop',
    version='0.3',
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.5",
        "python-socketio==4.5.1",
        "requests",
        "python-engineio==3.13.0"
    ]
)
