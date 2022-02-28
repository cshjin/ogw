from setuptools import find_packages, setup

setup(name="ogw",
      version="0.0.1",
      author="XXX",
      summary="TBD",
      license="MIT",
      packages=find_packages(exclude=["tests", "results", "log", "ogw.egg-info"]))
