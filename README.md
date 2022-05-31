

# Yolov5 Object Detection In OSRS using Python code, Detecting Cows - Botting

name: Executed with 4 FPS on MacBook Pro 2019 i7 3.1GHz CPU

# Quick Start

### installing python

installing python programming language

<a href="https://www.python.org/downloads/" rel="nofollow">python downloads website</a>

Click the download button, for the latest python version (for older version of Windows 7 or earlier, MacOS/Linux or Ubuntu use the other links).


### installing pycharm

installing pycharm <a href="https://www.jetbrains.com/pycharm/download/" rel="nofollow">pycharm</a>.

<a href="https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC" rel="nofollow">pycharm windows</a>


### Check your cuda version

type in terminal: ```nvidia-smi```

![image](https://user-images.githubusercontent.com/81003470/147712277-5b1fae1d-33b2-4ff0-a4de-19ef762e1b14.png)

I used CPU for training, however, GPU will perform better. CUDA 10.2 or up is needed.

Check if your gpu will work: https://developer.nvidia.com/cuda-gpus and use the cuda for your model and the latest cudnn for the cuda version.

full list of cuda versions: https://developer.nvidia.com/cuda-toolkit-archive

cuda 10.2 = https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe

### Install Cudnn

cuDNN = https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-10

### Download and Use LabelImg for creating annotated (object identified bounding boxes) image data

labelImg = https://tzutalin.github.io/labelImg/
for MAC users I recommend RectLabel on the appstore

## Install Module Requirements

in the terminal type:

```pip install -r requirements.txt```


## Check cuda version is compatiable with torch and torchvision

goto website and check version https://download.pytorch.org/whl/torch_stable.html

To take advantage of the gpu and cuda refer to the list for your cuda version search for cu<version of cuda no spaces or fullstops> e.g cu102 for cuda 10.2.
  
use the latest versions found, i at this point in time found: torch 1.9.0 and torchvision 0.10.0 (these 2 module versions so far i have had no issues other versions i get errors when running 
  
  in the terminal type the torch version + your cuda version (except for torchaudio no cuda version required):
  
  ```pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html```
  
  ![image](https://user-images.githubusercontent.com/81003470/147749033-c5de2a74-5365-444c-93c1-f5d9f75512c4.png)

  ```pip install torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html```
  
  ![image](https://user-images.githubusercontent.com/81003470/147749284-9411be6f-f000-4bf9-a167-b0d214b977f5.png)

  ```pip install torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```
  
- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Make sure when installing torchvision it doesn't try to install another version due to incompatability, try to either find a later version of torch or use a downgraded version of torchvision. there could be issues if another torch version is installed but the cuda version doesn't align with your gpu.`

## Test pytorch and cuda work
  
in the project run, the output should result in the device used as cuda, and the tensor calculations should run without errors:
  
  ![image](https://user-images.githubusercontent.com/81003470/147753127-c97b0ce4-e9c6-49d4-a817-f9a71928e240.png)

  This will also download the yolov5 weight files:
  
  ![image](https://user-images.githubusercontent.com/81003470/147753307-5c3df94e-206b-4bac-8f2d-8a5e7301c010.png)

# Custom training setup with YAML
  
<p><a href="https://www.kaggle.com/ultralytics/coco128" rel="nofollow">COCO128</a> is an example small tutorial dataset composed of the first 128 images in <a href="http://cocodataset.org/#home" rel="nofollow">COCO</a> train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of overfitting. <a href="https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml">data/coco128.yaml</a>, shown below, is the dataset config file that defines 1) the dataset root directory <code>path</code> and relative paths to <code>train</code> / <code>val</code> / <code>test</code> image directories (or *.txt files with image paths), 2) the number of classes <code>nc</code> and 3) a list of class <code>names</code>:</p>
  
```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]  # class names
  ```
Copying the method above i have done the same to format my data
  
```
# parent
# ├── yolov5
# └── datasets
#     └── osrs ← downloads here
#       └── crabs ← add each class
  #     └── xxx ← add each class


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./datasets/osrs  # dataset root dir
train: images/ # train images (relative to 'path') 128 images
val: images/  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['crab']  # class names

```

# Training

Epochs. Start with 200 epochs. If this overfits early then you can reduce epochs. If overfitting does not occur after 200 epochs, train longer, i.e. 600, 1200 etc epochs.

Image size. COCO trains at native resolution of --img 640, though due to the high amount of small objects in the dataset it can benefit from training at higher resolutions such as --img 1280. If there are many small objects then custom datasets will benefit from training at native or higher resolution. Best inference results are obtained at the same --img as the training was run at, i.e. if you train at --img 1280 you should also test and detect at --img 1280.

Batch size. Use the largest --batch-size that your hardware allows for. Small batch sizes produce poor batchnorm statistics and should be avoided.
In the terminal type:

Setting to train:

```python train.py --data osrs.yaml --weights yolov5s.pt --batch-size 2 --epoch 200``` 



## Training Finished

Once finished the resulting model best.pt and last.pt will be saved in the folder runs/train/exp<number>
  
![image](https://user-images.githubusercontent.com/81003470/147910872-6700f739-232e-42f4-a210-479dd7c12734.png)

# Detecting

This is where the detecting of objects take place, based on the parameters given, the code will run the default or custom weights and identify objects (inference) in 
images, videos, directories, streams, etc.
  
## Test Dectections

Run a test to ensure all is installed correctly, in the terminal type:

```python detect.py --source data/images/bus.jpg --weights yolov5s.pt --img 640```

![image](https://user-images.githubusercontent.com/81003470/148015379-5c099720-af00-425a-92b0-0d9e05545cd7.png)

This will run the default yolov5s weight file on the bus image and store the results in runs/detect/exp

These are the labels (the first integer is the class index and the rest are coordinates and bounding areas of the object)

```
5 0.502469 0.466204 0.995062 0.547222 # bus
0 0.917284 0.59213 0.162963 0.450926 # person
0 0.17284 0.603241 0.222222 0.469444 # person
0 0.35 0.588889 0.146914 0.424074 # person
```
 
Here is the resulting image with bounding boxes identifying the bus and people:
  
![bus](https://user-images.githubusercontent.com/81003470/148015666-65439829-1856-435f-a8d0-eea7b9baade0.jpg)

## Test Custom Detections (osrs cows model)

Move the trained model located in runs/train/exp<number> to the parent folder (overwrite the previous best.pt):

![image](https://user-images.githubusercontent.com/81003470/148020954-d42a32b0-b741-4791-8b69-300af762966d.png)
  
Let's see the results for osrs cow detection, to test in the terminal type:
  
``` python detect.py --source data/images/crabs.png --weights best.pt --img 640```

The labels results are:
```
  0 0.946552 0.362295 0.062069 0.0754098
0 0.398276 0.460656 0.106897 0.140984
0 0.426724 0.572131 0.105172 0.140984
0 0.352586 0.67377 0.122414 0.167213
0 0.310345 0.898361 0.117241 0.190164
0 0.151724 0.411475 0.062069 0.180328
0 0.705172 0.37541 0.0689655 0.127869
0 0.812931 0.319672 0.087931 0.127869
  ```
And here's the image result:
  
![image](https://user-images.githubusercontent.com/58753283/170961118-edf8254e-e4c4-4cdc-a5ef-944b423a649d.jpeg)

## Test Custom Detections (osrs cows model) on Monitor Display (screenshots with Pillow ImageGrab)
 
For a single screenshot, in the terminal type:

```python detect.py --source stream.jpg --weights best.pt --img 640 --use-screen ```

For a constant stream of the monitor display, in the terminal run:
  
```python detect_screenshots.py``` or right click on the detect_screenshots_only.py script and select run:
  
![image](https://user-images.githubusercontent.com/81003470/148022895-e34e65d6-0b6b-4d64-b9dc-a7b7ab38e148.png)

this will run the code with the default parameters listed below and can be changed to suit your needs.
  
```
def main_auto():
    run(weights='best.pt',  # model.pt path(s)
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        Run_Duration_hours=6,  # how long to run code for in hours
        Enable_clicks=False
        )
 ```

