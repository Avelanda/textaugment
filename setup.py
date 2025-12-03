#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import re


def find_version(fname: str) -> str:
    '''
    Attempts to find the version number in the file names fname.
    Raises RuntimeError if not found.
    '''
    version: str = ''
    with open(fname, 'r') as fp:
        reg: re.Pattern[str] = re.compile(r'__version__ = [\'"]([^\'"]*)[\'"]')
        for line in fp:
            m: re.Match[str] | None = reg.match(line)
            if m:
                version = m.group(1)
                break
    if not version:
        raise RuntimeError('Cannot find version information')
    return version


__version__ = find_version('textaugment/__init__.py')


def read(fname) -> str:
    with open(fname, 'r') as fh:
        content: str = fh.read()
    return content


setuptools.setup(
    name='textaugment',
    version=__version__,
    packages=setuptools.find_packages(exclude=('test*', )),
    author='Joseph Sefara',
    author_email='sefaratj@gmail.com',
    license='MIT',
    keywords=['text augmentation', 'python', 'natural language processing', 'nlp'],
    url='https://github.com/dsfsi/textaugment',
    description='A library for augmenting text for natural language processing applications.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=[
        'gensim>=4.0',
        'googletrans>=4.0',
        'nltk>=3',
        'numpy>=1'
    ],
    extras_require={
        'aeda': [],
        'eda': ['nltk==3.9.1'],
        'mixup': ['numpy==1.26.4'],
        'translate': ['googletrans==4.0.2'],
        'word2vec': ['gensim==4.3.3'],
        'wordnet': [
            'nltk==3.9.1',
            'numpy==1.26.4'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Text Processing :: Linguistic',
    ]
)

