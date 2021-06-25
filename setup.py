from setuptools import setup

setup(
    name="mypackage",
    version="0.0.1",
    packages=["mypackage"],
    install_requires=[
        "requests",
        'importlib; python_version == "2.6"',
    ],
)
