# MegaPose - Taskboard
This repository shows how to train simple MaskRCNN detector for the taskboard (or any custom object) using only synthetic data, and then run the MegaPose inference. For more information about the MegaPose, see the [HappyPose](https://github.com/agimus-project/happypose/tree/dev) repository.
## 1. Installation

### Create conda environment
```
conda env create -f environment.yml
conda activate happypose-det
export HAPPYPOSE_DATA_DIR="$PWD/local_data"
```
### Download MegaPose models
```
python -m happypose.toolbox.utils.download --megapose_models
```

## 2a. Download the pretrained detector
You can download the checkpoint for the pretrained taskboard detector from [here](https://drive.google.com/file/d/11p4mGXH0Vd2jl9fxLEH5qvy5_ebsDXdT/view?usp=sharing) and unzip it to `local_data/experiments`.

Alternatively, you can retrain it using the following steps.
## 2b. Detector training 
First, dowload BOP dataset meshes (YCB-V, T-LESS) with
```
sh download_bop_meshes.sh
```
This will download zipped 3D meshes for [YCB-V](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip) and [T-LESS](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tless/tless_models.zip) and unzip them to `local_data/bop_datasets/ycbv/` and `local_data/bop_datasets/tless/` respectively. The meshes will be used as distractor objects in the rendering.

Then, download blenderproc cc_textures and render the training synthetic data:
```
blenderproc download cc_textures local_data/cctextures
blenderproc run render_data.py
```

Finally, train the detector with
```
python -m scripts.train_detector
```

## 3. Run inference on example
Run the inference on the `taskboard-1` example:
```
python -m scripts.run_inference_on_example taskboard-1 --run-inference --vis-detections --vis-poses
```
The input data (image, mesh, camera intrinsics) are located in `local_data/examples/taskboard-1`.
