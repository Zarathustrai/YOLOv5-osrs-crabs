
import datetime
import os
import random
import pyautogui
from PIL import ImageGrab
import argparse
import sys
import time
from pathlib import Path

import cv2

import torch
import torch.backends.cudnn as cudnn

# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0

FILE = Path(__file__).absolute()
print(FILE)
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

# Set monitor size to capture
# monitor = (0, 0, 766, 530)

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, colorstr, non_max_suppression, \
    scale_coords, set_logging, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync

def GRABMSS_screen():
    #im = ImageGrab.grab(bbox=monitor) # left , top , right, bottom
    im = ImageGrab.grab()  # left , top , right, bottom
    im.save('fullscreen.png')
    im.close()
    return 'fullscreen.png'

def click_object(box):
    x = int(box[0])
    y = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    print('| x:', x, '| y:', y )
    d = random.uniform(0.01,0.05)
    #pyautogui.moveTo(round((x+x2)/2,0), round((y+y2)/2,0), duration=d) # center click
    pyautogui.moveTo(round((x + x2) / 2, 0), round((y + (y*0.1)), 0), duration=d)  # 10% upper (head) click
    d = random.uniform(0.01,0.05)
    pyautogui.click(button='left', duration=d)

@torch.no_grad()
def run(weights='best.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        Run_Duration_hours=6, # how long to run code for in hours
        Enable_clicks=True # will click on the class objects when confidence is greater than 90%
        ):
    agnostic_nms = False,  # class-agnostic NMS
    augment = False,  # augmented inference
    visualize = False,  # visualize features
    half = False,  # use FP16 half-precision inference
    t_end = time.time() + (60 * 60 * Run_Duration_hours)
    # using the datetime.fromtimestamp() function
    date_time = datetime.datetime.fromtimestamp(t_end)
    print(date_time)
    global fps, display_time, start_time

    # Initialize
    set_logging()
    device = 'cpu'
    half = False  # half precision only supported on CUDA

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    cudnn.benchmark = True  # set True to speed up constant image size inference
    time_clicked = time.time()
    # ------ attempt loop here ------
    while time.time() < t_end:
        source = GRABMSS_screen()
        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        # Run inference
        #model(torch.zeros(1, 3, *[1280,1280]).to(device).type_as(next(model.parameters())))
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            # Inference
            t1 = time_sync()
            pred = model(img, augment=augment, visualize=visualize)[0]
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()
            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                if len(det) == 0:
                    break
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(str(increment_path(save_dir / 'labels' / f'{p.stem}_full_{frame}.txt', mkdir=True).with_suffix('.txt')), 'w+') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                f.close()
                        if save_crop:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            cv2.imwrite(str(increment_path(save_dir / 'crops' / f'{p.stem}_full_{frame}.jpg', mkdir=True).with_suffix('.jpg')), imc)
                    if time.time() > time_clicked and float(conf) > 0.9 and Enable_clicks:
                        click_object(xyxy)
                        time_clicked = time.time() + 10
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    # cv2.destroyAllWindows()
                    cv2.imshow('aimbot', im0)
                    cv2.waitKey(1)

        os.remove("fullscreen.png")
        print(f'Done. ({time.time() - t0:.3f}s)')
        fps += 1
        TIME = time.time() - start_time
        if (TIME) >= display_time:
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()


def main_auto():
    run(weights='best.pt',  # model.pt path(s)
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
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

if __name__ == "__main__":
    main_auto()
