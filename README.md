# Files
- core
  - homography.py: related to homography warping
  - nn_utils.py: network related staffs, can be reused
  - model*.py: network, have the same interface
- data: dataloaders
- utils
- plot_pfm.py: plot a pfm file
- train.py
- val.py
- test.py

# Requirements
pytorch 1.4
opencv-python
apex: optional, comments the related code in train.py if not installed

# Notes
- dataset_name and model_name are used to dynamically load dataloader and model. it should be identical to the filename. 
- max_d and interval_scale are the same with MVSNet, cas_interv_scale are relative with interval_scale. e.g. the depth interval in stage i is (depth interval in cam file) * interval_scale * cas_interv_scale[i]
- num_samples = num_steps * batch_size
- the model is stored in save_dir/job_name, which should be put as load_path. load_path = -1 means to load the latest checkpoint
- torch.nn.functional.grid_sample has bug. align_corner option is reversed. 
