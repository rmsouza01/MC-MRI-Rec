from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# NOTE: Requirements file pending
# with open("requirements.txt") as f:
#     reqs = f.read()

setup(
    name="mrrec",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    # install_requires=reqs.strip().split("\n"),
    author="Dr. Roberto Souza",
    description="Utilities for Multi-Channel MR Image Reconstruction Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=r"https://github.com/rmsouza01/MC-MRI-Rec",
    classifiers=["Programming Language :: Python :: 3.8"],
    test_suite="pytest",
)
