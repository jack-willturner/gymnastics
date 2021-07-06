import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gymnastics",
    version="0.0.1",
    author="Jack Turner",
    author_email="jackwilliamturner@icloud.com",
    description="A lightweight toolkit for neural architecture search experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jack-willturner/gymnastics/",
    project_urls={
        "Bug Tracker": "https://github.com/jack-willturner/gymnastics/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
