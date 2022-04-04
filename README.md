# C3M-new
A reimplementation of the CoRL'20 paper "[Learning Certified Control Using Contraction Metric](https://arxiv.org/abs/2011.12569)", by Dawei Sun, Susmit Jha, and Chuchu Fan. Code is based on the [origin repo](https://github.com/MIT-REALM/ccm). Reimplemented by [Songyuan Zhang](https://syzhang092218-source.github.io).

## Dependencies
You need to have the following libraries with [Python3](https://www.python.org/):

- [NumPy](https://numpy.org/)
- [Gym](https://gym.openai.com/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [descartes](https://pypi.org/project/descartes/)
- [shapely](https://github.com/shapely/shapely)
- [tqdm](https://tqdm.github.io/)
- [PyYAML](https://pyyaml.org)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

To install requirements:

```bash
conda create -n c3m python=3.8
conda activate c3m
pip install -r requirements.txt
```

## Run

### Available environments
- DubinsCarTracking

Creating a new tracking environment is pretty easy. You can create a new file in `model/env`, and create a new class as a child of `.base.TrackingEnv`. Then you just need to re-implement all the abstract methods and properties. 

### Training
You can train the C3M using the following command:

```bash
python train_c3m.py --env DubinsCarTracking
```

This will create a new folder with time stamp in the `log/[ENV]` folder, with `models`, `summary`, and `settings.yaml` inside. The trained models will be saved in `models`, the training log will be saved in `summary`, and the settings for training will be saved in `settings.yaml`.

Here are some options for training:

- `--env`: The environment to be trained in.
- `--n-iter`: Number of training iterations.
- `--batch-size`: Sample batch size of the states.
- `--no-cuda`: Disable cuda.

### Testing
After training, you can easily test the controller using:

```bash
python test_c3m_policy.py --path [PATH-TO-THE-LOG]
```

Note that `[PATH-TO-THE-LOG]` means the path to the folder with time stamp. This will print the reward of each episode, and create a new `videos` folder in `[PATH-TO-THE-LOG]`, containing the animations. 

If you want to see the log of the training process, use:
```bash
tensorboard --logdir [PATH-TO-THE-SUMMARY]
```

## Acknowledgement
Thank [Dawei Sun](https://www.daweisun.me) for the origin code and [Glen Chou](https://web.eecs.umich.edu/~gchou/) for the helpful discussion on the training process. 
