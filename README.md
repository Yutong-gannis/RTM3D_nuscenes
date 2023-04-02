# RTM3D

## Installation

## Conda Environment

#### 1. Create Conda Env
```bash
conda create -n capstone python=3.8 numpy cudatoolkit=10.2
```

#### 2. Install PyTorch and Torchvision
Instalasi PyTorch dan Torchvision dapat mengikuti dua cara tergantung dari GPU yang digunakan. Untuk GPU terkini atau yang dipakai pada notebook dapat menginstall langsung melalui `pip`:
```bash
pip install torch==1.9.0 torchvision==0.10
```
Jika menggunakan GPU lama (Cuda Compute < 3.5), dapat mendownload `.whl` terlebih dahulu pada blog [Nelson Liu](https://cs.stanford.edu/~nfliu/files/pytorch/whl/torch_stable.html). Setelahnya dapat melakukan installasi melalui pip:
```bash
pip install ./torch-1.9.0+cu102-cp38-cp38-linux_x86_64.whl
pip install ./torchvision-0.10.0+cu102-cp38-cp38-linux_x86_64.whl
```

#### 3. Install Requirements
Package/library lainnya yang diperlukan dapat diinstal melalui `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 4. Compile Deformable Convolution (DCNv2)

Clone DCNv2 for PyTorch > 1.8 from [ruhyadi/DCNv2_18](https://github.com/ruhyadi/DCNv2_18)
```bash
cd src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/ruhyadi/DCNv2_18 ./DCNv2
```

Build DCNv2 from scrach
```bash
cd DCNv2
python setup.py build develop
```

#### 5. Compile IoU3D
Clone IoU3D from [ruhyadi/iou3d](https://github.com/ruhyadi/iou3d)
```
cd src/lib/utils
rm -rf iou3d
git clone https://github.com/ruhyadi/iou3d ./iou3d
```
Build IoU3D from scract
```
cd iou3d
python setup.py install
```
Failed on PyTorch 1.9.0, but can inference model.

### Docker
Docker image can be build with:
```bash
docker build -t username/rtm3d:latest .
```

Or you can pull docker image (**recommend**) with:
```bash
docker pull -t ruhyadi/rtm3d:latest
```

Then you can run docker container in interactive mode with:
```bash
./runDocker.sh
```

## Dataset Preparation

### Convert nuScenes to KITTI
```
python src/tools/export_nuscenes.py
```

### Generate Image List
```
python src/tools/create_sets_nuscenes.py
```

### Convert nuScenes to COCO
```
python src/tools/nuscenes_to_coco.py
```

## Inference
```bash
python ./src/faster.py \
    --demo ./demo_kitti_format/data/kitti/image \
    --calib_dir ./demo_kitti_format/data/kitti/calib/ \
    --load_model ./weights/model_res18_1.pth \
    --gpus 0 \
    --vis \
    --arch res_18
```
