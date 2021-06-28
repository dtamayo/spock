from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

classifier_requirements = [
    'rebound>=3.14.0', 'scikit-learn', 'xgboost>=1.1.0'
]
regression_requirements = [
    'matplotlib', 'pytorch_lightning>=1.0.0', 'torch>=1.5.1', 'torchvision',
    'scipy', 'rebound>=3.14.0', 'scikit-learn', 'einops', 'matplotlib', 'numpy',
    'pandas'
]
analytical_requirements = [
    'rebound>=3.14.0', 'numpy', 'celmech'
]

regression_models = [
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_0_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_10_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_11_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_12_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_13_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_14_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_15_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_16_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_17_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_18_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_19_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_1_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_20_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_21_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_22_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_23_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_24_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_25_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_26_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_27_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_28_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_29_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_2_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_3_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_4_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_5_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_6_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_7_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_8_output.pkl',
'models/regression/steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v50_9_output.pkl'
]

exec(open('spock/version.py').read())
setup(name='spock',
    version=__version__,
    description='Stability of Planetary Orbital Configurations Klassifier',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy astrophysics exoplanets stability',
    packages=['spock'],
    package_data={'spock': ['models/featureclassifier.json'] + regression_models},
    install_requires=list(set(classifier_requirements + regression_requirements + analytical_requirements)),
    tests_require=["numpy"],
    test_suite="spock.test",
    zip_safe=False)
