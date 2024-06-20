#!/usr/bin/sh

mkdir -p local_data/bop_dataset
cd local_data/bop_datasets
mkdir ycbv
mkdir tless

cd ycbv
wget https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip
unzip ycbv_models.zip

cd ../tless
wget https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/tless/tless_models.zip
unzip tless_models.zip

