from setuptools import setup, find_packages

setup(
    name="synthEHRella",
    version="1.0.0",
    author="Xingran Chen, Zhenke Wu, Xu Shi, Hyunghoon Cho, Bhramar Mukherjee",
    author_email="chenxran@umich.edu",
    description="SynthEHRella: A Package for Synthetic EHR Data Generation Benchmarking",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/chenxran/synthEHRella",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'pickle',
    ],
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'synthehr=synthEHRella.run_generation:main',
        ],
    },
)
