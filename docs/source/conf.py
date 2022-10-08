# Configuration file for the Sphinx documentation builder.

# -- Project information
import pyg_multiagent
import datetime


project = 'pytorch_geometric_multiagent'
author = 'Chenning Yu'
copyright = f'{datetime.datetime.now().year}, {author}'

# release = '0.1'
# version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "myst_parser",
]

html_theme = 'pygma_sphinx_theme'
html_title = "PyG-MultiAgent Documentation"
html_logo = ('https://raw.githubusercontent.com/rainorangelemon/pygma_sphinx_theme/'
             'master/pygma_sphinx_theme/static/img/pygma_logo.png')
html_favicon = ('https://raw.githubusercontent.com/rainorangelemon/pygma_sphinx_theme/'
                'master/pygma_sphinx_theme/static/img/favicon.png')

add_module_names = False
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'pytorch-geometric': ('https://pytorch-geometric.readthedocs.io/en/latest', None),
}

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
# }
# intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# # -- Options for HTML output

# # -- Options for EPUB output
# epub_show_urls = 'footnote'

def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'pyg_multiagent': pyg_multiagent}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect('source-read', rst_jinja_render)
