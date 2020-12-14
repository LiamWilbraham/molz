from setuptools import setup

description = "Molecular analysis tool"
long_description = """
Statistical analysis tool to help identify molecular fragments that promote, or detract from,
target properties.

Sepecifically, this tool calculates the "z-scores" of molecular substructures in a given
sub-population of a database to identify fragments that are over- or under-represented in this
sub-population relative to a reference population. These substructures can either be specified
by the user, or automatically generated using Morgan fingerprints.
"""

setup(name='molz',
      version='0.1.2',
      description=description,
      long_description=long_description,
      url='https://github.com/LiamWilbraham/molz',
      author='Liam Wilbraham',
      author_email='liam.wilbraham@glasgow.ac.uk',
      license='MIT',
      packages=['molz'],
      zip_safe=False)
