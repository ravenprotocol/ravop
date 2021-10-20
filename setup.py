from setuptools import setup, find_packages

setup(
    name='ravop',
    version='0.2-alpha',
    packages=find_packages(),
    install_requires=[
        "SQLAlchemy_Utils==0.37.2",
        "numpy==1.20.1",
        "redis==3.5.3",
        "SQLAlchemy==1.3.23",
        "gevent_socketio==0.3.6"
    ],
    dependency_links=[
    ]
)
