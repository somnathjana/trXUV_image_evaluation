# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:27:08 2021

@author: jana
"""

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='trXUV_image_evaluation',
    version='1.0',
    packages=['trXUV_image_evaluation'],
    url='https://github.com/somnathjana/trXUV_image_evaluation',  # Optional
    install_requires=['numpy', 'matplotlib', 'scipy', 'h5py'],  # Optional
    license='MIT',
    author='Somnath Jana',
    author_email='sj.phys@gmail.com',
    description='Evaluation of the image data @ HHG-Lab, MaxBornInstitute-B1-Berlin',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
)