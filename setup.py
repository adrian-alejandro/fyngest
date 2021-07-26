# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fyngest',
    version='0.1.0',
    description='fyngest is a financial data ingestion tool',
    long_description=readme,
    author='Adrián Zelaya',
    author_email='zelaya.adrian@gmail.com',
    url='https://github.com/adrian-alejandro/fyngest',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
