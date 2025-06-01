
# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, ToTensor, Resize
from iouEval import iouEval, getColorEntry

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_binary_tensor(tensor):
    # If input is a torch tensor, convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    plt.imshow(tensor, cmap='gray')  # 0=black, 1=white
    plt.axis('off')
    plt.show()




seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--method', default='msp', choices=['msp', 'maxlogit', 'entropy'],
                    help="Anomaly scoring method: msp, maxlogit, or entropy")
    
    parser.add_argument('--model',default="erfnet",choices=['erfnet', 'enet', 'bisenet'])
    
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits (used in MSP)')
    parser.add_argument('--void', action='store_true', help='Use Void class for anomaly detection')


    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:/semester 3/AML/project/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.void :
        if args.method == 'msp' :
            print("✅ Void setup activated")
        else :
            print("❌ On void setup you should set the method to msp")
            sys.exit(1)


    path_list = [
        "D:/semester_3/AML/project/datasets/RoadAnomaly21/images/*.png",
        "D:/semester_3/AML/project/datasets/RoadObsticle21/images/*.webp",
        "D:/semester_3/AML/project/datasets/FS_LostFound_full/images/*.png",
        "D:/semester_3/AML/project/datasets/fs_static/images/*.jpg",
        "D:/semester_3/AML/project/datasets/RoadAnomaly/images/*.jpg"
    ]

    # if not os.path.exists('results.txt'):
    #     open('results.txt', 'w').close()
    # file = open('results.txt', 'a')


    if args.model == "bisenet" :
        weightspath = '../trained_models/Checkpoint20_20.pth'
    elif args.model == "erfnet" :
        weightspath = '../trained_models/erfnet_pretrained.pth'
    elif args.model == "enet" :
        weightspath = '../save/ENet_Cityscapes/ENet'


    print ("Loading weights: " + weightspath)

    if args.model == "erfnet" :
        print("model is erfnet")
        model = ERFNet(NUM_CLASSES)
    elif args.model == "enet" :
        print("model is enet")
        model = ENet(NUM_CLASSES)
    elif args.model == "bisenet" :
        print("model is bisenet")
        model = BiSeNetV2(NUM_CLASSES)


    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        missing = []
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                elif 'module.'+ name in own_state.keys():
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
        state_dict = checkpoint['model_state_dict']
    elif args.model == 'erfnet' :
        state_dict = checkpoint

    model = load_my_state_dict(model, state_dict)
    print ("Model and weights LOADED successfully")

    IGNORE_INDEX = 2
    PERCENTILE = 95

    iouEvalVal_overall = iouEval(nClasses = 3, ignoreIndex = IGNORE_INDEX )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for current_path in path_list :
        
        iouEvalVal_dataset = iouEval(nClasses = 3, ignoreIndex = IGNORE_INDEX )

        print("args.input before globbing:", current_path)

        dataset = ''
        if 'FS_LostFound_full' in current_path :
            dataset = 'FS_LostFound_full'
        elif 'fs_static' in current_path :
            dataset = 'fs_static'
        elif 'RoadAnomaly21' in current_path :
            dataset = 'RoadAnomaly21'
        elif 'RoadAnomaly' in current_path :
            dataset = 'RoadAnomaly'
        elif 'RoadObsticle21' in current_path :
            dataset = 'RoadObsticle21'


        model.eval()

        if "*" in current_path or "?" in current_path:  
            expanded = glob.glob(current_path)
        else:
            expanded = [current_path]


        print("Expanded files:")
        for f in expanded:
            print(f, "| isfile:", os.path.isfile(f))

        # Only keep actual image files
        valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        current_path = [
            f for f in expanded 
            if f.lower().endswith(tuple(valid_extensions)) and os.path.isfile(f)
        ]

        print(f"✅ Found {len(current_path)} valid image(s) to process.")



        if len(current_path) == 0:
            print("❌ No images found! Please check the --input path.")
            exit(1)
            # for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        for path in current_path :

            print(path)
            images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
            images = images.permute(0,3,1,2)


            # print("images shape ",images.shape)
            with torch.no_grad():
                result = model(images)
                
            if isinstance(result, tuple):
                result = result[0]


            logits = result.squeeze(0).data.cpu().numpy()


            if args.method == "msp" :
                temperature = getattr(args, 'temperature', 1.0)

                if args.void :
                    if args.model == 'erfnet' :
                      softmax_probs = F.softmax(result / temperature, dim=1)  # Shape: [B, C, H, W]
                      void_prob = softmax_probs[:, 19, :, :] 
                      void_prob_np = void_prob.squeeze(0).data.cpu().numpy()
                      anomaly_result = void_prob_np
                    elif args.model == 'bisenet' or args.model == 'enet' :
                        softmax_probs = F.softmax(result / temperature, dim=1)  # Shape: [B, C, H, W]
                        void_prob = softmax_probs[:, 0, :, :] 
                        void_prob_np = void_prob.squeeze(0).data.cpu().numpy()
                        anomaly_result = void_prob_np
                        
                  
                else :
                  softmax_probs = F.softmax(result / temperature, dim=1)  # Shape: [B, C, H, W]
                  msp, predicted = torch.max(softmax_probs, dim=1)  # Shape: [B, H, W]
                  anomaly_result = 1 - msp
                  anomaly_result = anomaly_result.squeeze(0).data.cpu().numpy()

                
                # static threshold
                # anomaly_result = (anomaly_result - anomaly_result.min()) / (anomaly_result.max() - anomaly_result.min() + 1e-8)
                # threshold = 0.5
                # binary_map = (anomaly_result > threshold).astype(np.uint8) 

                # percentile based
                threshold = np.percentile(anomaly_result, PERCENTILE)
                binary_map = (anomaly_result > threshold).astype(np.uint8)


            elif args.method == "maxlogit" :
                # MaxLogit anomaly score
                anomaly_result = -np.max(logits, axis=0)  # shape: (H, W)
                threshold = np.percentile(anomaly_result, PERCENTILE)
                binary_map = (anomaly_result > threshold).astype(np.uint8)

            elif args.method == "entropy" :
                # logits = result.squeeze(0).data.cpu().numpy()  # shape: (C, H, W)
                # # Softmax (numerically stable)
                # exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
                # softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                # # Compute entropy
                # entropy = -np.sum(softmax * np.log(softmax + 1e-12), axis=0)  # shape: (H, W)
                # anomaly_result = entropy


                # `result` is your logits tensor of shape [1, 20, 720, 1280]
                # 1) Convert logits to probabilities and log-probabilities along the class axis
                probs     = F.softmax(result,    dim=1)  # → [1, 20, 720, 1280]
                log_probs = F.log_softmax(result, dim=1)  # → [1, 20, 720, 1280]

                # 2) Compute entropy at each pixel: H = −∑ p⋅log p over classes
                entropy_map = -torch.sum(probs * log_probs, dim=1)  # → [1, 720, 1280]

                # 3) (Optional) Normalize by log(C) if you want values in [0,1]:
                C = result.size(1)  # =20
                entropy_map = entropy_map / torch.log(torch.tensor(C, dtype=entropy_map.dtype))

                # 4) Squeeze batch dim and convert to NumPy if needed
                anomaly_result = entropy_map.squeeze(0).cpu().numpy()  # → [720, 1280]

                threshold = np.percentile(anomaly_result, PERCENTILE)
                binary_map = (anomaly_result > threshold).astype(np.uint8)


            # visualize_binary_tensor(binary_map)


            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT:
                pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
                pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT:
                pathGT = pathGT.replace("jpg", "png")  

            mask = Image.open(pathGT)
            ood_gts = np.array(mask)

            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)



            ood_gts[ood_gts == 255] = IGNORE_INDEX

            binary_map = torch.from_numpy(binary_map).unsqueeze(0).unsqueeze(0).to(device).long()
            ood_gts = torch.from_numpy(ood_gts).unsqueeze(0).unsqueeze(0).to(device).long()


            ood_gts_np = ood_gts.cpu().numpy()


            if 1 not in np.unique(ood_gts_np):
                continue              
            else:
              iouEvalVal_overall.addBatch(binary_map, ood_gts)
              iouEvalVal_dataset.addBatch(binary_map, ood_gts)
            del result, anomaly_result, ood_gts, mask
            torch.cuda.empty_cache()      

        iouVal_dataset, iou_classes_dataset = iouEvalVal_dataset.getIoU()
        iouStr = getColorEntry(iouVal_dataset)+'{:0.2f}'.format(iouVal_dataset*100) + '\033[0m'
        print (f"MEAN IoU for dataset {dataset}: {iouStr}% ")
    iouVal, iou_classes = iouEvalVal_overall.getIoU()


    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print (f"Overall MEAN IoU based on {args.method} with temp {args.temperature}: {iouStr}%")


if __name__ == '__main__':
    main()