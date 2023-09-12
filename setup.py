from setuptools import setup

setup(
   name='nodesplat',
   version='0.1.0',
   author='ddrous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['nodesplat'],
   url='http://pypi.python.org/pypi/nodesplat/',
   license='LICENSE.md',
   description='Parallel-in-tme gaussian splatting with physics',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
         "jax >= 0.3.4",
         "optax >= 0.1.1",
         "seaborn",
   ],
)
