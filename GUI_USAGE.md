[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)

## GUI DEMO CODE USAGE

### License
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


### Setup

- Upgrade pytorch to 0.4.0 (required by the segmentation network code).
  - `conda install pytorch=0.4.0 torchvision cuda91 -c pytorch`
- Pull the CSAIL segmentation network from this fork 
  - `git submodule add https://github.com/mingyuliutw/semantic-segmentation-pytorch segmentation`
- Run the demo code to download the network and make sure the environment is set up properly. 
  - `cd segmentation` 
  - `./demo_test.sh`
- Install gensim, an NLP library
  - `conda install -c anaconda gensim 
`
- Download GoogleNews-vectors-negative300.bin.gz (link https://code.google.com/archive/p/word2vec/); unzip and store under segmentation/  

- Construct semantic mapping 
  - `python semantic_matching.py`
  - The semantic distance output is stored in **semantic_rel.npy**
  
- Install PyQt5
  - `conda install -c dsdale24 pyqt5`
  
- Put the folder segmenatation/ in the PYTHONPATH variable

- run `python gui.py`