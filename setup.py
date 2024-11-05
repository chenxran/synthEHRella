from setuptools import setup, find_packages

setup(
    name="synthEHRella",  # Name of your package
    version="0.1.0",  # Initial version
    author="Xingran Chen, Zhenke Wu, Bhramar Mukherjee",  # Your name or organization
    author_email="chenxran@umich.edu",  # Your contact email
    description="A benchmarking package for synthetic EHR data generation and evaluation",
    long_description=open('README.md').read(),  # This will be the content of your README file
    long_description_content_type='text/markdown',  # Format of README (can be text/plain or text/markdown)
    url="https://github.com/chenxran/SynthEHRella",  # URL to the project (GitHub link, for example)
    packages=find_packages(),  # Automatically find all the packages in your project
    # install_requires=[
    #     'pandas',       # Example dependencies, you can list more as needed
    #     'numpy',
    #     'omegaconf',
    #     'tqdm',
    # ],
    # classifiers=[
    #     'Programming Language :: Python :: 3',  # Ensure it works with Python 3+
    #     'License :: OSI Approved :: MIT License',  # License type
    #     'Operating System :: OS Independent',  # OS compatibility
    # ],
    python_requires='>=3.7',  # Minimum Python version required
    entry_points={
        'console_scripts': [
            'synthehr=synthEHRella.run_generation:main',  # For running the script from the command line
        ],
    },
)