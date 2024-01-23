# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['xm_solarlib',
 'xm_solarlib.bifacial',
 'xm_solarlib.ivtools',
 'xm_solarlib.pvfactors',
 'xm_solarlib.pvfactors.geometry',
 'xm_solarlib.pvfactors.irradiance',
 'xm_solarlib.pvfactors.viewfactors',
 'xm_solarlib.spa_c_files',
 'xm_solarlib.spectrum']

package_data = \
{'': ['*']}

install_requires = \
['cython>=3.0.4,<4.0.0',
 'ephem>=4.1.5,<5.0.0',
 'h5py==3.9.0',
 'importlib>=1.0.4,<2.0.0',
 'matplotlib==3.7.2',
 'numba>=0.58.1,<0.59.0',
 'numpy==1.25.2',
 'pandas==2.0.3',
 'pytest-mock>=3.12.0,<4.0.0',
 'scipy==1.11.2',
 'shapely>=1.6.4.post2,<2',
 'solarfactors>=1.5.3,<2.0.0']

setup_kwargs = {
    'name': 'xm-solarlib',
    'version': '0.1.0',
    'description': '',
    'long_description': '# XM Solar Library\n\n## Configuración base\n\n### Activar librería XM Solar\n\n```source $(poetry env info -p)/bin/activate```\n\n### Instalación de paquetes\n\n```poetry install```\n\n### Adición de paquete\n\n```poetry add <package>```\n\n### Verificar path poetry env\n\n```poetry env info```\n\nejemplo: */home/{user}/.cache/pypoetry/virtualenvs/xm-solarlib-C5Mlu42b-py3.10*\n\n### Correr test\n\n```poetry run pytest```\n',
    'author': 'Jhonatan Acelas Arevalo',
    'author_email': 'jhonatanacelas@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.13',
}


setup(**setup_kwargs)