from setuptools import find_packages, setup

setup(
    name='industry-solutions-release',
    version='1.0',
    author='Antoine Amend',
    author_email='antoine.amend@databricks.com',
    description='Deploy solution accelerators as HTML files',
    include_package_data=True,
    install_requires=[
        'databricks-api==0.9.0',
    ],
    long_description_content_type='text/markdown',
    url='https://github.com/databricks-industry-solutions/industry-solutions-release',
    packages=find_packages(where='.'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: Other/Proprietary License',
    ],
)