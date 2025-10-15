


# **S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction**

### Accelerate reconstruction pipeline for large-scale dynamic street scene.

<p align="center">
    🖥️ <a href="https://github.com/Tom-zgt/S3R-GS">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/pdf/2503.08217">Paper </a> &nbsp&nbsp 
<br>

## Requirements

Codes was tested with the following dependencies

- Python 3.10
- CUDA 11.8
- PyTorch 2.0.1

## Installation

1. Clone the repository

```
git clone https://github.com/Tom-zgt/S3R-GS.git
```

2. Create a new conda environment as [map4d](https://github.com/tobiasfshr/map4d.git)

```
conda create --name map4d -y python=3.10
conda activate s3rgs
pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/hturki/tiny-cuda-nn.git@ht/res-grid#subdirectory=bindings/torch
pip install nerfstudio==1.0.3
python setup.py develop
```

## Pepare Data

Use our preprocessing scripts to prepare the datasets:

```
mkdir data
#put dataset in the folder ./data/
```

- Pepare dataset [Argoverse 2](https://github.com/tobiasfshr/map4d/blob/main/docs/datasets/Argoverse2.md) as [map4d](https://github.com/tobiasfshr/map4d.git)
- Download the preprocessed KITTI scenes [here](https://drive.google.com/file/d/15lJSoaNPbvhrkHGTpOTGRjHZewtd6f1B/view?usp=sharing)

## Training

Use the generated metadata files to train the model on a specific dataset:

```
ns-train 4dgf-kitti street --data data/KITTI/tracking/training/metadata_[0001/0002/0006]_cover.pkl --train-split-fraction [0.75/0.5/0.25]

ns-train 4dgf-kitti-800 [(optional) --machine.num-devices 4] street --data data/KITTI/tracking/training/metadata_[0009/0020].pkl --train-split-fraction 0.75

ns-train 4dgf-av2-big --machine.num-devices 8 --pipeline.model.max-num-gaussians 8000000 --pipeline.model.object-grid-log2-hashmap-size 17 street --data data/Argoverse2/metadata_PIT_6180_1620_6310_1780.pkl --voxel-size 0.15
ns-train 4dgf-av2-big --machine.num-devices 8 --pipeline.model.max-num-gaussians 8000000 --pipeline.model.object-grid-log2-hashmap-size 17 street --data data/Argoverse2/metadata_PIT_1100_-50_1220_150.pkl --voxel-size 0.15
```

## Evaluation

We provide the trained checkpoints  [here](https://drive.google.com/file/d/1Qkj04HddA5P5e4JeJgsXr3rOvMQkCvGV/view?usp=sharing)

```
ns-eval --load-config <trained_model_config>
```

## Citation

Please consider citing our work with the following references

```
@article{zheng2025s3r,
  title={S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction},
  author={Zheng, Guangting and Deng, Jiajun and Chu, Xiaomeng and Yuan, Yu and Li, Houqiang and Zhang, Yanyong},
  journal={arXiv preprint arXiv:2503.08217},
  year={2025}
}
```

## Acknowledgements

This project builds on the great work [map4d](https://github.com/tobiasfshr/map4d.git). 
