import sys
import os


sys.path.insert(0, os.path.abspath("../../"))

project = "pysyncon"
copyright = "2024, Stiofán Fordham"
author = "Stiofán Fordham"
release = "1.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
]
html_theme = "alabaster"
bibtex_bibfiles = ["biblio.bib"]
