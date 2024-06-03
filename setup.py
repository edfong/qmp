from setuptools import setup,find_packages
#May need to install Pystan separately with pip
setup(name='qmp',
      version='1.0.0',
      description='Quantile Martingale Posterior',
      url='http://github.com/edfong/qmp',
      author='Edwin Fong',
      author_email='chefong@hku.hk',
      license='BSD 3-Clause',
      packages=find_packages(),
      install_requires=[
          'numpy==1.26.4',
          'scipy==1.12.0',
          'scikit-learn',
          'pandas',
          'matplotlib',
          'seaborn',
          'tqdm',
          'jax==0.4.25',
          'jaxlib==0.4.25',
      ],
      include_package_data=True,
      python_requires='>=3.7'
      )
