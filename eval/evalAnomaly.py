# Copyright (c) OpenMMLab. All rights reserved.
import os
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
    parser.add_argument('--method', default='msp', choices=['msp', 'maxlogit', 'entropy'],
                    help="Anomaly scoring method: msp, maxlogit, or entropy")
    
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits (used in MSP)')
 
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:/semester 3/AML/project/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    print("args.input before globbing:", args.input)

    dataset = ''
    if 'FS_LostFound_full' in args.input :
        dataset = 'FS_LostFound_full'
    elif 'fs_static' in args.input :
        dataset = 'fs_static'
    elif 'RoadAnomaly21' in args.input :
        dataset = 'RoadAnomaly21'
    elif 'RoadAnomaly' in args.input :
        dataset = 'RoadAnomaly'
    elif 'RoadObsticle21' in args.input :
        dataset = 'RoadObsticle21'


    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()

    if "*" in args.input or "?" in args.input:  
        expanded = glob.glob(args.input)
    else:
        expanded = [args.input]


    print("Expanded files:")
    for f in expanded:
        print(f, "| isfile:", os.path.isfile(f))

    # Only keep actual image files
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    args.input = [
        f for f in expanded 
        if f.lower().endswith(tuple(valid_extensions)) and os.path.isfile(f)
    ]

    print(f"✅ Found {len(args.input)} valid image(s) to process.")



    if len(args.input) == 0:
        print("❌ No images found! Please check the --input path.")
        exit(1)
        # for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    for path in args.input :
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
            ##### first I tried this #####
            # anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
            ##### first I tried this #####


            logits = result.squeeze(0).data.cpu().numpy()


            if args.method == "msp" :

                temperature = getattr(args, 'temperature', 1.0)
                # Apply temperature scaling to logits
                scaled_logits = logits / temperature
                # Softmax over channel dimension (axis=0)
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=0, keepdims=True))  # for stability
                softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                anomaly_result = 1.0 - np.max(softmax, axis=0)
                print("result shape", anomaly_result.shape)


            

            elif args.method == "maxlogit" :
                logits = result.squeeze(0).data.cpu().numpy()  # shape: (C, H, W)
                # MaxLogit anomaly score
                anomaly_result = -np.max(logits, axis=0)  # shape: (H, W)

            elif args.method == "entropy" :
                logits = result.squeeze(0).data.cpu().numpy()  # shape: (C, H, W)
                # Softmax (numerically stable)
                exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
                softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                # Compute entropy
                entropy = -np.sum(softmax * np.log(softmax + 1e-12), axis=0)  # shape: (H, W)
                anomaly_result = entropy


        
   
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

    print("val_label shape:", val_label.shape)
    print("val_label values:", val_label.flatten()[:10])

    print("val_out shape:", val_out.shape)
    print("val_out values:", val_out.flatten()[:10])


    prc_auc = average_precision_score(val_label, val_out)

    print("Label unique values and counts:", np.unique(val_label, return_counts=True))


    if np.sum(val_label) == 0:
        print("No positive labels in validation set — skipping FPR@95")
    else:
        fpr = fpr_at_95_tpr(val_label, val_out)


    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write((dataset + '     ' + args.method + '     ' + str(args.temperature) + '    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()


if __name__ == '__main__':
    main()