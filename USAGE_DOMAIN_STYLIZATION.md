[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)

## CODE USAGE

### License
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


### Setup
In addition to the step in [USAGE.md](USAGE.md), you need to install anaconda opencv3
- `conda install -c menpo opencv3`
- `pip install opencv-contrib-python`

### Perform Domain Stylization

`python process_list.py --fast --outp_img_folder [YOUR_PATH] `
`                       --cont_img_folder [YOUR_PATH] --cont_list [YOUR_PATH] --cont_seg_folder [YOUR_PATH OPTIONAL]\`
`                       --styl_img_folder [YOUR_PATH] --styl_list [YOUR_PATH] --styl_seg_folder [YOUR_PATH OPTIONAL]\`
