from setuptools import setup, find_packages
setup(
  name = 'FADiff',         # How you named your package folder (MyLib)
  packages=find_packages(),   # Chose the same as "name"
  version = '0.19',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'autodiff using forward and reverse mode',   # Give a short description about your library
  author = 'teamxvii',                   # Type in your name
  
  url = 'https://github.com/liuxiaoxuan97/FADiff',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/liuxiaoxuan97/FADiff/archive/v_19.tar.gz',    # I explain this later on
  keywords = ['autodiff', 'forward mode', 'reverse mode'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pytest',
          'coverage',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
