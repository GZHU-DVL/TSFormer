# TSFormer: Tracking Structure Transformer for Image Inpainting

Jiayu Lin and Yuan-Gen Wang

## Preparation
This repository is built in PyTorch 1.8.1 and tested on environment (Python3.7, CUDA11.1, nvcc11.1).

1.Preparing the environment:
```
conda create -n TSFormer python=3.7
conda activate TSFormer
```

2.check nvcc version
```
nvcc -V
```
If there is no nvcc or the version does not correspond, then install the corresponding version of nvcc.

3.Install dependencies
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm timm tensorboardX

pip install einops gdown addict future lmdb numpy pyyaml pyaml requests scipy tb-nightly yapf lpips ninja albumentations
```
## Datasets
### Image Dataset
We evaluate the proposed method on three public datasets, including [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder#6-paris-street-view-dataset) and [Places2](http://places2.csail.mit.edu/), which are widely adopted in the literature.

**CelebA.**
Since the image size of CelebA dataset is 178 x 218. According to the general experimental operation, crop the center of the image to an image size of 178 x 178 and then scale it to 256 x 256.

Modify transform.py
```
cd /datasets/transform.py
```

**Paris StreetView and Places2.** 

Scale the image size to 256 x 256.

**Mask Dataset.**
Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

## Training
Since our model runs on four 24GB GPUs, we support distributed training. You can train model in distributed settings.
```
python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --image_root [path to image directory] \
  --mask_root [path mask directory]
```
Otherwise, you can also change the configuration file in the options directory.

## Testing

To test the model, you run the following code:
```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```

Pre-trained models can be obtained from [Google Cloud Disk](https://drive.google.com/drive/folders/1RTNRF31EasqqphUNbgWYz4_Q7YjGlXqa?usp=sharing).

## Evaluation

The diversity can be evaluated by:
```
python scripts/metrics/cal_psnr.py  --gt_dir [path to groundtruth directory] --result_dir [path to output directory]
python scripts/metrics/cal_lpips.py  --path1 [path to groundtruth directory] --path2 [path to output directory] --device cuda:0
```
Otherwise, you can also change the configuration file in the scripts/metrics directory.

## Multi-Scale Image Inpainting

1.Modify transform.py
```
cd /datasets/transform.py
```
2.run test_splice.py
```
python test_splice.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```

## License

This source code is made available for research purpose only.
