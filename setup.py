from setuptools import find_packages, setup

__version__ = '0.0.2'
URL = 'https://github.com/rainorangelemon/pytorch_geometric_multiagent'

install_requires = [
    'tqdm',
    'numpy',
]

full_requires = [
    'tqdm',
    'numpy',
    'matplotlib',
]

benchmark_requires = [
    'wandb',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='pyg_multiagent',
    version=__version__,
    description='Graph Neural Network Library for Multi-Agent',
    author='Chenning Yu',
    author_email='rainorangelemon@gmail.com',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'deep-learning',
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'pytorch-geometric',
        'multi-agent'
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'full': full_requires,
        'benchmark': benchmark_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
