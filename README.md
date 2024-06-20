# Installation

## Create conda environment
```
conda env create -f environment.yml
conda activate happypose-det
export HAPPYPOSE_DATA_DIR="$PWD/local_data"
```

## Download data
### BOP datasets (YCB-V, T-LESS)
Either run
```
sh download_data.sh
```
or manually downlod zipped 3D models for [YCB-V](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip) and [T-LESS](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tless/tless_models.zip) and unzip them to `local_data/bop_datasets/ycbv/` and `local_data/bop_datasets/tless/` respectively.

### Blenderproc cc_textures
```
blenderproc download cc_textures local_data/cctextures
```

## Render synthetic data
```
blenderproc run render_data.py
```


