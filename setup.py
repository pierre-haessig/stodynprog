from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

setup(

    name = "stodynprog",
    version = '0.1',
    packages = ['stodynprog'],

    test_suite='stodynprog.tests',

    cmdclass = {'build_ext': build_ext},
    ext_modules = [

        Extension(
		    'stodynprog.dolointerpolation.multilinear_cython',
		    ['stodynprog/dolointerpolation/multilinear_cython.pyx'],
            extra_compile_args=['-O3']
        ),

    ],

    install_requires = ["numpy","scipy","cython"],

    extras_require = {
            'plots':  ["matplotlib"],
    },

    author = "Pierre Haessig",
    author_email = "pierre.haessig@crans.org",

    description = 'a library for solving stochastic optimal control problems, also called dynamic optimization',

    license = 'BSD-3',

    url = 'https://github.com/pierre-haessig/stodynprog',

)
