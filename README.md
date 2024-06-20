# Installation

## 1. Create conda environment
```
conda env create -f environment.yml
conda activate happypose-det
export HAPPYPOSE_DATA_DIR="$PWD/local_data"
```

## 2. Download data
### Dowload BOP datasets (YCB-V, T-LESS)
```
sh download_data.sh
```
This will download zipped 3D models for [YCB-V](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip) and [T-LESS](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tless/tless_models.zip) and unzip them to `local_data/bop_datasets/ycbv/` and `local_data/bop_datasets/tless/` respectively.

### Download blenderproc cc_textures
```
blenderproc download cc_textures local_data/cctextures
```

### Download MegaPose models
```
python -m happypose.toolbox.utils.download --megapose_models
```

## 3. Render synthetic data
```
blenderproc run render_data.py
```

## 4. Train the detector
```
python -m scripts.train_detector
```

## 5. Run inference on example
```
python -m scripts.run_inference_on_example taskboard-1  --run-inference --vis-detections --vis-poses
```