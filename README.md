# KCLA

##  Installation

### Requirements


- Python 3.8+
- PyTorch 2.1.2 
- MMDetection 




### Our environments
- OS: Ubuntu 22.04
- Python: 3.8
- CUDA: 12.2
- pytorch   2.1.2 
- GPU: RTX 4090
- mmcv   2.1.0    
- mmdet  3.3.0           
- mmengine  0.10.7                

## Usage

### Train with ResNet on CIFAR-10/CIFAR-100

You can start training on CIFAR-100 with the following command:

``` bash
 python cifar.py  -a kcla --dataset  cifar100   --train-batch 128 --wd 1e-4  --log-dir checkpoint/KCLA --depth 56 
```

### Train with ResNet on ImageNet-1k
You can start training on ImageNet-1k with the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py '/imagenet' -a resnet50_kcla -b 256 --epochs 100 --warmup-epochs 3 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:12300' --world-size 1 --rank 0 --workers 10
```

### Train with Faster R-CNN and Mask R-CNN on COCO2017
First, please configure the environment according to the official mmdetection documentation.

Put the resnet_kcla.py file in this repository into the mmdetection/mmdet/models/backbones/ folder. For example, put the resnet_dlal.py file into the './mmdetection/mmdet/models/backbones/' folder. Put the configuration file into the './mmdetection/configs/faster_rcnn/' folder. For example, put the faster_rcnn_r50dlal_fpn_1x_coco.py file into the './mmdetection/configs/faster_rcnn/' folder.

Then execute the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py configs/faster_rcnn/faster_rcnn_r50kcal_fpn_1x_coco.py --cfg-options data.samples_per_gpu=8
```


### Image segmentation experiment

- Before conducting image segmentation experiments, it is first necessary to download the three datasets: ISIC2017, ISIC2018, and Kvasir - SEG.

- After successfully downloading these datasets, for the convenience of subsequent experimental operations, they need to be placed in the './data' folder within the project directory. The following is a reference example of the file format, taking the ISIC17 dataset as an example:

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

- You can find the configuration files in the './configs' folder and adjust relevant hyperparameters and other settings.

- To train the model, please execute the following command:

```bash
python train.py
```