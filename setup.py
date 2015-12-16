from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np

setup(

    name = "stodynprog",
    version = '0.1.1',
    packages = ['stodynprog', 'stodynprog.dolointerpolation', 'stodynprog.tests'],

    test_suite='stodynprog.tests',

    cmdclass = {'build_ext': build_ext},
    ext_modules = [

        Extension(
		    'stodynprog.dolointerpolation.multilinear_cython',
		    ['stodynprog/dolointerpolation/multilinear_cython.pyx'],
            extra_compile_args=['-O3']
        ),

    ],
	include_dirs = [np.get_include()], #find numpy headers to build the Cython extension

    install_requires = ["numpy","scipy","cython"],

    extras_require = {
            'plots':  ["matplotlib"],
    },

    author = "Pierre Haessig",
    author_email = "pierre.haessig@crans.org",

    description = 'a library for solving stochastic optimal control problems, also called dynamic optimization',

    license = 'BSD-3',

    url = 'https://github.com/pierre-haessig/stodynprog',
    
    classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: BSD License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    # 'Programming Language :: Python :: 3',
    # 'Programming Language :: Python :: 3.2',
    # 'Programming Language :: Python :: 3.3',
    # 'Programming Language :: Python :: 3.4',
],

)
