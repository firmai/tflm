from setuptools import setup

setup(name='tflm',
      version='0.1',
      description='Transformations for Linear Model',
      url='https://github.com/firmai/tflm',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['tflm'],
      install_requires=[
          'pandas',
          'numpy',
          'lightgbm',
          'pandas',
          'shap',
          'keras',
          'sklearn'

      ],
      zip_safe=False)
