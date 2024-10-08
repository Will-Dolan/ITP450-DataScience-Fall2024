# Week 7

The first step is to install the necessary packages from RAPIDS package. First, get an interactive node:
```
salloc --partition=gpu --gres=gpu:1 -N 1 -n 8 --mem=32GB --time=1:00:00 --account=irahbari_1147 --reservation=itp-450-th
```

Then, we create the conda environment ([Installation Guide - RAPIDS Docs](https://docs.rapids.ai/install/))
```
conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia  \
    cudf=24.08 cuml=24.08 python=3.11 cuda-version=12.2
```

Activate the environment:
```
conda activate rapids-24.08
```

Install Jupyter and create the kernel:
```
mamba install -c conda-forge ipykernel matplotlib
python -m ipykernel install --user --name rapids-24.08 --display-name rapids-24.08
```
