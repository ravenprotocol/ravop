from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ravop',
    version='0.4.1',
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravftp',
    keywords='Ravop, requestor client library',
    install_requires=[
        "numpy==1.21.5",
        "python-socketio==5.4.1",
        "python-engineio==4.2.1",
        "requests"
        "python-dotenv"
    ]
)
