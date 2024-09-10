from setuptools import setup, find_packages

setup(
    name='embedding_buckting',   # Name of your package
    version='0.1.0',            # Version number
    packages=find_packages(),   # Automatically find all packages in the source folder
    install_requires=[
        'numpy==2.1.1',
        'openai==1.44.1',
        'scikit-learn==1.5.1'
    ],        # List your project dependencies here
    url='https://github.com/Rafipilot/embedding_buckting',  # URL of your GitHub repo
    author='Rafayel Latif',
    author_email='rafayel.latif@gmail.com',
    description='Bucketing using text embeddings',
    long_description=open('README.md').read(),  # Read the content of your README file
    long_description_content_type='text/markdown',  # Ensure the README file is in Markdown
    license='MIT',  # Specify the license
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Artificial Life',
    ],
)
