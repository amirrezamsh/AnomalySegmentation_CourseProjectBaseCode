# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2

import transforms as ext_transforms


NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()


enet_input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),              # → [0,1] RGB, no further normalization
])


enet_target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    ext_transforms.PILToLongTensor(),  # maps void→0, classes→1…19
])

erfnet_input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
erfnet_target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

# -------------

# mapping_20 = { 
#     0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
#     7: 1, 8: 2,
#     9: 0, 10: 0,
#     11: 3, 12: 4, 13: 5,
#     14: 0, 15: 0, 16: 0,
#     17: 6, 18: 0, 19: 7, 20: 8, 21: 9,
#     22: 10, 23: 11, 24: 12, 25: 13, 26: 14,
#     27: 15, 28: 16,
#     29: 0, 30: 0,
#     31: 17, 32: 18, 33: 19,
#     # -1: 0
#     255:19
# }

# # Create fast lookup table (for performance)
# mapping_array = np.full(256,19, dtype=np.uint8)
# for k, v in mapping_20.items():
#     mapping_array[k] = v

# # Transformation function for the label
# def pil_to_mapped_tensor(pic):
#     label = np.array(pic)  # Convert PIL to numpy
#     # label[label == -1] = 255
#     label = mapping_array[label]  # Apply mapping
#     return torch.from_numpy(label).long()


# bisenet_input_transform_cityscapes = Compose([
#     Resize((512, 1024), Image.BILINEAR),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
# ])


# bisenet_target_transform_cityscapes = Compose([
#     Resize((512, 1024), Image.NEAREST),
#     pil_to_mapped_tensor,
# ])


# -----------

bisenet_input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

def pil_to_long_tensor(pic):
    return torch.from_numpy(np.array(pic)).long()

bisenet_target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    pil_to_long_tensor,
])


def main(args):


    if args.model == "erfnet" :
        weightspath = args.loadDir + "erfnet_pretrained.pth"
    elif args.model == "enet" :
        weightspath = args.loadDir + "ENet"
    elif args.model == "bisenet" :
        weightspath = args.loadDir + "checkpoint20.pth"

    print ("Loading weights: " + weightspath)

    if args.model == "erfnet" :
        model = ERFNet(NUM_CLASSES)
    elif args.model == "enet" :
        model = ENet(NUM_CLASSES)
    elif args.model == "bisenet" :
        model = BiSeNetV2(NUM_CLASSES)


    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        missing = []
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                elif 'module.'+ name in own_state.keys() :
                    own_state['module.' + name].copy_(param)
                
                elif name not in own_state and f"module.{name}" not in own_state and name.split("module.")[-1] not in own_state:
                    missing.append(name)

            else:
                own_state[name].copy_(param)
        print(f"missing keys : {missing}")
        return model
    
    checkpoint = torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=False)

    if args.model == "enet" :
        state_dict = checkpoint['state_dict']

    elif args.model == 'bisenet' :
        model = load_my_state_dict(model, checkpoint['model_state_dict'])

    elif args.model == 'erfnet' :
        model = load_my_state_dict(model, checkpoint)


    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    if args.model == 'enet' : 
        loader = DataLoader(cityscapes(args.datadir, enet_input_transform_cityscapes, enet_target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    elif args.model == 'erfnet' :
        loader = DataLoader(cityscapes(args.datadir, erfnet_input_transform_cityscapes, erfnet_target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    elif args.model == "bisenet" :
        loader = DataLoader(cityscapes(args.datadir, bisenet_input_transform_cityscapes, bisenet_target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)




    # iouEvalVal = iouEval(NUM_CLASSES)
    iouEvalVal = iouEval(nClasses = NUM_CLASSES )


    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()


        print("filename : ",filename)


        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]


        if args.model == 'enet' :
            # Prepare labels
            labels = labels.unsqueeze(1)  # shape: (B,1,H,W)

            # Map labels: 255 (void) → 19 (ignore index), 1–19 → 0–18
            labels[labels == 255] = 0

            # Same remap for preds if model trained on 1–19
            preds = preds - 1
            preds[preds == -1] = 19  # match label remapping

        if args.model == 'bisenet' :
            labels = labels.unsqueeze(1)
            labels[labels == 255] = 19
            
            preds = preds - 1
            preds[preds == -1] = 19  # match label remapping


        # Feed to IoU computation
        iouEvalVal.addBatch(preds, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--model',default="erfnet")

    parser.add_argument('--loadDir',default="../trained_models/")
    # parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:\semester_3\AML\project\datasets\cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
