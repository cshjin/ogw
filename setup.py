from setuptools import find_packages, setup

setup(name="ogw",
      version="0.0.1",
      author="Hongwei Jin",
      summary="Orthogonal Gromov-Wasserstein Distance",
      license="MIT",
      packages=find_packages(exclude=["tests", "results", "debug", "log", "ogw.egg-info"]))
