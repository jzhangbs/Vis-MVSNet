# Visibility-aware Multi-view Stereo Network
## Introduction
## How to Use
### Environment Setup
The code is tested in the following environment. The newer version of the packages should also be fine. 
```
python==3.7.6
apex==0.1                # only for sync batch norm
matplotlib==3.1.3        # for visualization in val.py and test.py
numpy==1.18.1
opencv-python==4.1.2.30
open3d==0.9.0.0          # for point cloud I/O
torch==1.4.0
tqdm==4.41.1             # only for the progressbar
```
It is highly recommended to use Anaconda. 

You need to install `apex` manually. See https://github.com/NVIDIA/apex for more details. 

### Data preparation
Download the [Blended low res set](https://drive.google.com/open?id=1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb), [Tanks and Temple testing set](https://drive.google.com/open?id=1YArOJaX9WVLJh4757uE8AEREYkgszrCo). For more information, please visit [MVSNet](https://github.com/YoYo000/MVSNet). 

The pre-processed DTU dataset will be available soon.

### Training, validation & testing
First set the machine dependent parameters e.g. dataset dir in `sh/dir.json`.

Set the job name, and run `python sh/bld.py local` or `python sh/dtu.py local` to train the network on BlendedMVS/DTU. 

Set the job name to load and the number of sources, and run `python sh/bld_val.py local` or `python sh/dtu_val.py local` to validate the network on BlendedMVS/DTU. 

Set the dataset dir, dir of the models, job name to load and the output dir, and run `sh/tnt.sh` or `sh/dtu.sh` to generate the outputs for point cloud fusion on Tanks and Temples/DTU. (Note that the indexing of your shell should start from 0, otherwise you need to modify the scripts.)

See `python train.py/val.py/test.py --help` for the explanation of all the flags.

### Explanation of depth number and interval scale
`max_d` and `interval_scale` is a standard depth sampling. Similar to MVSNet, in the preprocessing, `depth_start` is kept, `depth_interval` is scaled by `interval_scale`, and `depth_num` is set to be `max_d`. So if you want to keep the depth range in the cam files, to need to manually ensure `max_d*interval_scale=<depth num in the cam file>`

`cas_depth_num` and `cas_interv_scale` are used in the coarse-to-fine architecture. The number in `cas_interv_scale` is applied to the depth interval __after__ the preprocessing. As is mentioned in the paper, the first stage consider the full depth range. So the parameters are manually set as `depth_num = 256 = 64*4 = cas_depth_num*cas_interv_scale`.

### Post-Processing
The code for point cloud fusion will be available soon.

## File Formats
The format of all the I/O files follows MVSNet, except for that we output three probability maps instead of one. 

## Changelog
### Aug 19 2020
- Add README
- Add train/val/test scripts