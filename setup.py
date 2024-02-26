from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='Handy package for training output constrained neural networks',
    long_description=open('README.md').read(),
    install_requires=['dependency1', 'dependency2'],
    url='https://github.com/yourusername/your_package_name',
    author='Jannick Stranghöner',
    author_email='jannick.stranghoener@iosb-ina.fraunhofer.de'
)