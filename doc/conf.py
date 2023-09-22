# -*- coding: utf-8 -*-

import sys, os

sys.path.insert(0, os.path.abspath('extensions'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

todo_include_todos = True
templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = []
add_function_parentheses = True
#add_module_names = True
# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

project = u'fracture-suite'
copyright = u'2023, Leon Bohmann'

version = '0.1.6'
release = ''

# -- Options for HTML output ---------------------------------------------------

html_title = "Fracture Suite"
#html_short_title = None
#html_logo = None
#html_favicon = None
html_static_path = ['_static']
html_domain_indices = False
html_use_index = False
html_show_sphinx = False
htmlhelp_basename = 'MusicforGeeksandNerdsdoc'
html_show_sourcelink = False