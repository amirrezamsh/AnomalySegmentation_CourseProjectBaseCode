# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F

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
    parser.add_argument(
    "--input",
    default="D:/semester_3/AML/project/datasets/RoadAnomaly21/images/*.png",
    help="Glob pattern to match images"
)
    parser.add_argument('--model',default="erfnet")

    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:/semester 3/AML/project/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()


    path_list = [
        "D:/semester_3/AML/project/datasets/RoadAnomaly21/images/*.png",
        "D:/semester_3/AML/project/datasets/RoadObsticle21/images/*.webp",
        "D:/semester_3/AML/project/datasets/FS_LostFound_full/images/*.png",
        "D:/semester_3/AML/project/datasets/fs_static/images/*.jpg",
        "D:/semester_3/AML/project/datasets/RoadAnomaly/images/*.jpg"
    ]

    if not os.path.exists('results_void.txt'):
        open('results_void.txt', 'w').close()
    file = open('results_void.txt', 'a')


    loadModel = ''

    if args.model == "erfnet" :
        weightspath = '../trained_models/erfnet_pretrained.pth'
        loadModel = 'ErfNet'
    elif args.model == "enet" :
        weightspath = '../trained_models/ENet'
        loadModel = 'ENet'
    elif args.model == 'bisenet' :
        weightspath = '../trained_models/Checkpoint20_20.pth'
        loadModel = 'BiseNet'


    print ("Loading model: " + loadModel)
    print ("Loading weights: " + weightspath)

    if args.model == "erfnet" :
        model = ERFNet(NUM_CLASSES)
    elif args.model == "enet" :
        model = ENet(NUM_CLASSES)
    elif args.model == 'bisenet' :
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
    model.eval()

    for current_path in path_list :

        anomaly_score_list = []
        ood_gts_list = []

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
            with torch.no_grad():
                result = model(images)

                if isinstance(result, tuple):
                    result = result[0]

                logits = result.squeeze(0).data.cpu().numpy()  # shape: (C, H, W)
        

                if args.model == 'erfnet' :
                      softmax_probs = F.softmax(result, dim=1)  # Shape: [B, C, H, W]
                      void_prob = softmax_probs[:, 19, :, :] 
                      void_prob_np = void_prob.squeeze(0).data.cpu().numpy()
                      anomaly_result = void_prob_np
                elif args.model == 'bisenet' or args.model == 'enet' :
                    softmax_probs = F.softmax(result , dim=1)  # Shape: [B, C, H, W]
                    void_prob = softmax_probs[:, 0, :, :] 
                    void_prob_np = void_prob.squeeze(0).data.cpu().numpy()
                    anomaly_result = void_prob_np


                
    
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

            if 1 not in np.unique(ood_gts):
                continue              
            else:
                ood_gts_list.append(ood_gts)
                anomaly_score_list.append(anomaly_result)
            del result, anomaly_result, ood_gts, mask
            torch.cuda.empty_cache()

        file.write( "\n")

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))


        prc_auc = average_precision_score(val_label, val_out)



        if np.sum(val_label) == 0:
            print("No positive labels in validation set — skipping FPR@95")
        else:
            fpr = fpr_at_95_tpr(val_label, val_out)


        print(f'AUPRC score: {prc_auc*100.0}')
        print(f'FPR@TPR95: {fpr*100.0}')

        file.write((args.model + '     ' + dataset + '     ' + '    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()


if __name__ == '__main__':
    main()