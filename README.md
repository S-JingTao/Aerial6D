## Aerial6D
Aerial6D: Towards Aerial Category-Level Pose Tracking for Air-Ground Robotic Manipulation

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
