import sys
import os


sys.path.insert(0, os.path.abspath("../../"))

project = "pysyncon"
copyright = "2023, Stiofáin Fordham"
author = "Stiofáin Fordham"
release = "0.2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax"
]
html_theme = "alabaster"
