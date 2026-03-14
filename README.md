## Aerial6D
Aerial6D: Towards Aerial Category-Level Pose Tracking for Air-Ground Robotic Manipulation

## Create Environment

```bash
conda create -n aerial6d python==3.8
conda activate aerial6d
```
## Download Dataset
[NOCS-REAL275](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)
[Wild6D](https://drive.google.com/drive/folders/1SjWUcuSvYMM5rPPd4aQhK0jo1IHbCJbT)
[HouseCat6D](https://sites.google.com/view/housecat6d)
[PhoCal](https://www.campar.in.tum.de/public_datasets/2022_cvpr_wang/README.html)
[Omni6DPose](https://jiyao06.github.io/Omni6DPose/download)
## Requirements

* Python 3.8
* PIL
* scipy
* numpy
* logging
* CUDA 11.8

## Training

Please run:
```bash
python train.py
```

## Evaluation

Please run:
```
python eval.py
```

After generating the estimated pose results of each frame, please run:

```
python benchmark.py
```
## Inference
After preparing the data and fill in some information blanks in `inference.py`, please run:
```sh
python inference.py
```
