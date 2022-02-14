from setuptools import setup, find_packages

setup(
    name='ravop',
    version='0.3',
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.5",
        "python-socketio==5.4.1",
        "python-engineio==4.2.1",
        "requests"
    ]
)
