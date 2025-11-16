"""Setup script for pg-renderer package."""

from setuptools import setup, find_packages

setup(
    name="pg-renderer",
    version="0.1.0",
    description="Pure Python PG (Problem Generation) renderer for WeBWorK",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
)

