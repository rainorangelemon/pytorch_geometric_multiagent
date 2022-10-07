# Configuration file for the Sphinx documentation builder.

# -- Project information
import pyg_multiagent
import datetime


project = 'pytorch_geometric_multiagent'
author = 'Chenning Yu'
copyright = f'{datetime.datetime.now().year}, {author}'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

html_theme = 'furo'
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

# templates_path = ['_templates']

# # -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'

# # -- Options for EPUB output
# epub_show_urls = 'footnote'

def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'pyg_multiagent': pyg_multiagent}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect('source-read', rst_jinja_render)
