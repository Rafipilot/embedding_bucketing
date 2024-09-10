from setuptools import setup, find_packages

setup(
    name='embedding_buckting',   # Name of your package
    version='0.1.0',            # Version number
    packages=find_packages(),   # Automatically find all packages in your source folder
    install_requires=[],        # List your project dependencies here
    url='https://github.com/Rafipilot/embedding_buckting',  # URL of your GitHub repo
    author='Rafayel Latif',
    author_email='rafayel.latif@gmail.com',
    description='Bucketing using text embeddings'
)