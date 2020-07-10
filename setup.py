from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('spock/version.py').read())
setup(name='spock',
    version=__version__,
    description='Stability of Planetary Orbital Configurations Klassifier',
    long_description=long_description,
    url='https://github.com/dtamayo/spock',
    author='Daniel Tamayo',
    author_email='tamayo.daniel@gmail.com',
    license='GPL',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy astrophysics exoplanets stability',
    packages=['spock'],
    package_data={'spock': ['models/featureclassifier.json']},
    install_requires=["rebound", "sklearn", "xgboost>=1.1.0"],
    tests_require=["numpy"],
    test_suite="spock.test",
    zip_safe=False)
