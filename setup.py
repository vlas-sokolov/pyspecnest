import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return open(os.path.join(here, fname)).read()

setup(
    name = "pyspecnest",
    version = "0.0.0",

    author = "Vlas Sokolov",
    author_email = "vlas145@gmail.com",

    description = ("Nested sampling of pyspeckit models with PyMultiNest."),
    long_description=read('README.md'),

    license = "MIT",
    url = "https://github.com/vlas-sokolov/pyspecnest",

    packages = ["pyspecnest"],
)
