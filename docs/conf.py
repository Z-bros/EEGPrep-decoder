"""Sphinx configuration for the EEGPrep-decoder documentation.

Build locally::

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

import os
from importlib.metadata import version as _version

project = "EEGPrep-decoder"
author = "Zidane Fatuna"
copyright = "2026, Zidane Fatuna"
release = _version("eeg-decoder")
version = release

extensions = [
    "myst_nb",                   # Markdown pages and notebooks (includes myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",                  # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
myst_heading_anchors = 3

# Notebooks are stored without outputs and executed on every build, so an
# example that stops working fails the (warnings-as-errors) build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
numpydoc_show_class_members = False   # the autosummary tables list members already
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False

# EEGPREP_DOCS_OFFLINE=1 skips the external links (for builds without network).
intersphinx_mapping = {} if os.environ.get("EEGPREP_DOCS_OFFLINE") else {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Only the API reference has sub-pages; elsewhere the left sidebar would be empty.
html_sidebars = {page: [] for page in (
    "index", "install", "guide/*", "examples/*", "module-notes", "mne-behaviours",
    "validation", "benchmarks", "ROADMAP")}
html_title = f"EEGPrep-decoder {release}"
html_theme_options = {
    "github_url": "https://github.com/Z-bros/EEGPrep-decoder",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {
    "github_user": "Z-bros",
    "github_repo": "EEGPrep-decoder",
    "github_version": "main",
    "doc_path": "docs",
}
