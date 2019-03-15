from setuptools import setup

setup(name='efars',
      version='0.1',
      description='',
      url='',
      author='Leonard Herold',
      license='MIT',
      packages=['efars'],
      install_requires=[
          "predictionio",
          "requests",
          "docker",
          "python-dotenv",
          "numpy",
          "pandas",
          "matplotlib"
      ],
      zip_safe=False)