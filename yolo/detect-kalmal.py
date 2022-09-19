# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync





@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # weigh.pt 불러오기
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # 웨이트의 확장자 확인하기
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #웹캠을 제외한 모든 소스파일 불러오기

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size







    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # 추론 시작
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    # dt[0] : 0-1사이 값으로 정규화에 걸린시간
    # dt[1] : 추론에 걸린 시간
    # dt[2] :  nonmaxsuppression에 걸린 시간

    # 칼만 필터 불러오기
    KF = KalmanFilter((1/ dataset.frames) , 0, 0, 1, 0.01 , 0.01)


    # Obeject Dection 여부 확인을 위한 텐서
    Zero_pred = torch.zeros((0, 6), device='cuda')
    zero_whcc = torch.zeros((1, 4), device='cuda').view(-1)
    Last_whcc = 0.5 * torch.ones((1, 4), device='cuda').view(-1)
    Last_x_y = [[0], [0]]

    x_list= []
    y_list= []
    for path, img, im0s, vid_cap in dataset:

        # path : Path
        # im0s : 원본값
        # img : 원본값 / 255
        # vid_cap : cv2.VideoCapture()
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0 (0~1 값으로 정규화)

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 추론
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # save_dir : 저장하는 위치
            pred = model(img, augment=augment, visualize=visualize)[0]

            # pred shape : [1,15120,6]



        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        #bbox가 겹치는 부분을 제게해주는 과정
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 검출된 오브젝트의 숫자 i
        # det : pred의  i번째 원소
        # 검출됬을때 텐서는 리스트에 저장됨
        # [tensor([[2.49784e+02, 3.63123e+01, 4.66263e+02, 3.21688e+02, 2.71220e-01, 0.00000e+00]], device='cuda:0')]
        # 검출 됐을때 텐서

        # Kalman Predict
        (X, Y) = KF.predict()


        li = []
        # Kalman Update
        for i, d in enumerate(pred):

            #yolo모델이 검출에 실패한 경우
            if torch.equal(d, Zero_pred) and not (torch.equal(zero_whcc, Last_whcc)):

                #칼만필터로 예측된 값을 텐서 변환
                X = (torch.tensor(X, device="cuda")).view(-1)
                Y = (torch.tensor(Y, device="cuda")).view(-1)

                #예측 좌표와 이전 프레임의 값을 가져와 (1,6)의 텐서로 결합
                d = torch.cat([X, Y, Last_whcc], dim=0)

                #텐서의 차원 맞추기
                d = d.view(1,-1)

            # 칼만 혹은 yolo에서 예측된 텐서를 보정
            for *xyxy, conf, clas in reversed(d):
                # xywh 형태로 변형
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # List 형태로 반환
                print(xywh)

                # 텐서의 shape 맞추기
                conf = conf.view(-1)
                clas = clas.view(-1)

                # 지난 프레임의 예측 값을 측정값으로 사용하여
                # 칼만필터의 추정값을 계산
                est_x_y = KF.update(Last_x_y)

                # 칼만필터에 input 형태로 변형
                Last_x_y = [[xywh[0]], [xywh[1]]]

                # 칼만필터의 추정값을 텐서에 적용
                xywh[0] = est_x_y[0, 0]
                xywh[1] = est_x_y[1, 0]

                # 높이와 너비값 저장
                wh = (torch.tensor(xywh[2:4], device="cuda"))
                Last_whcc = torch.cat([wh, conf, clas], dim=0)

                # Yolo의 출력 형태로 텐서 길이로 변경
                xywh = [xywh]
                xyxy = (xyxy2xywh(torch.tensor(xyxy, device="cuda").view(1, 4))).view(-1)
                # 원래 형태로 만들어주기
                Esti_X = torch.cat([xyxy, conf, clas], dim=0)

                # 텐서를 차원을 맞추어 리스트에 저장

                Esti_X = Esti_X.unsqueeze(0)
                li.append(Esti_X)

        if li is not None:
            pred =[sum_tensor(li)]
        """
        
         칼만 예측이 없는 모델은 데이터가 저장되는 숫자가 다름 
         여기가 아니라 detect에서 수정

        """


        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            #modelc : pretrain된 resnet
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process predictions
        # i : 인덱스 번호
        # 2개의 물체가 검출되면 i = 0, 1
        # det : pred의  i번째 원소[conf, y2, x2, y1, x1, 0]
        #각각의 캡쳐된 이미지마다
        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                # p : Path
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0) #dataset에서 frame 가져오기

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            # annotator : 주석달기

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # c : detection된 class 이름
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # s : 클래스가 각각 몇개 검출됬는지 저장해주는 문자열

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xywh
                        # xywh[:, 0] = x center
                        # xywh[:, 1] = y center
                        # xywh[:, 2] = width
                        # xywh[:, 3] = height

                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)#cv2.CAP_PROP_FPS : 프레임의 속도
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))#프레임의 너비
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#프레임의 높이
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
