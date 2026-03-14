## Aerial6D
Aerial6D: Towards Aerial Category-Level Pose Tracking for Air-Ground Robotic Manipulation

## Create Environment

```bash
conda create -n aerial6d python==3.8
conda activate aerial6d
```
## Download Dataset
[NOCS-REAL275](https://github.com/S-JingTao/Aerial6D)
[Wild6D](https://github.com/S-JingTao/Aerial6D)
[HouseCat6D](https://github.com/S-JingTao/Aerial6D)
[PhoCal](https://github.com/S-JingTao/Aerial6D)
[Omni6DPose](https://github.com/S-JingTao/Aerial6D)
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
