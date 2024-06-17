# Installation

## Create conda environment
```
conda env create -f environment.yml
conda activate happypose-det
```

## Download data
### Blenderproc cc_textures
```
blenderproc download cc_textures cctextures
```
### BOP datasets (YCB-V, T-LESS)
Downlod zipped models for [YCB-V](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip) and [T-LESS](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tless/tless_models.zip) and unzip them to `bop_datasets/ycbv/` and `bop_datasets/tless/` respectively.

## Render synthetic data
```
blenderproc run blenderproc run render_custom.py
```


