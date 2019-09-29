from setuptools import setup, find_packages

setup(name='dl_rank',
      version='0.3.3.3',
      description='Packaging tensorflow for Recommend on local and EMR',
      author='wangqi@696',
      author_email='qiwang@clubfactory.com',
      url='https://github.com/Snoworday/dl_rank',
      packages=find_packages(), #['dl_rank', 'dl_rank.model', 'dl_rank.conf'],
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
          "License :: OSI Approved :: MIT License",
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          "Operating System :: OS Independent"
      ]
      )

