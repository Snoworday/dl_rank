from setuptools import setup, find_packages

setup(name='dl_rank',
      version='0.1',
      description='Deeplearning Package for Recommend onlocal and EMR',
      author='wangqi@696',
      author_email='qiwang@clubfactory.com',
      packages=find_packages(
          exclude=['tensorflowonspark']), #['dl_rank', 'dl_rank.model', 'dl_rank.conf'],
      # package_dir={'':'dl_rank'},
      package_data={
          'conf': ['I2Iconf_uv/*']
      },
      zip_safe=False,
      install_requires=[
          'tensorflow==1.13.1',
          'pandas>=0.24.2',
          'paramiko>=2.6.0',
          'pyarrow==0.14.1'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7    u',
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"
      ]
      )

