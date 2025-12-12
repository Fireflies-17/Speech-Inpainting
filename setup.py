from setuptools import setup, find_packages

setup(name='Speech Inpainting',
      version='0.2.0',
      description='Encoder and Decoder adaptation',
    #   license='MIT',
      install_requires=open("requirements.txt", "r").read().splitlines(),
      keywords=[
          'Speech',
          'Deep Learning',
      ])

