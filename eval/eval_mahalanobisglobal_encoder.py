# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
import joblib

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
from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import torch.nn.functional as F
from train.main import MyCoTransform
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import cv2
from sklearn.decomposition import PCA


import torch
import numpy as np

import torch
import numpy as np


input_transform = Compose([
    Resize((512, 1024), interpolation=Image.BILINEAR),
    ToTensor()
])

target_transform = Compose([
    Resize((512, 1024), interpolation=Image.NEAREST)
])


def mahalanobis_score(features, mean, cov, pca_model=None):

    # Apply PCA if provided
    if pca_model is not None:
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = pca_model.transform(features)

    # Convert everything to torch tensors
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32)
    if not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov, dtype=torch.float32)

    # Move mean and cov to the same device as features
    device = mean.device
    features = features.to(device)

    inv_cov = torch.inverse(cov)

    # Compute Mahalanobis score
    delta = features - mean
    scores = torch.einsum('nd,dd,nd->n', delta, inv_cov, delta)
    return scores



def visualize_anomaly_detection(input_image, mahalanobis_map, gt_mask, title=None):
    """
    input_image: PIL Image or tensor [3, H, W]
    mahalanobis_map: tensor [H, W] — normalized [0, 1]
    gt_mask: numpy array or tensor [H, W] — 1 for OOD, 0 for in-dist
    """
    # Convert input image to displayable format if it's a tensor
    if isinstance(input_image, torch.Tensor):
        if input_image.dim() == 3:
            input_image = TF.to_pil_image(input_image.cpu())
        else:
            raise ValueError("Expected 3D tensor for input image")

    # Convert tensors to numpy arrays
    if isinstance(mahalanobis_map, torch.Tensor):
        mahalanobis_map = mahalanobis_map.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    # Create figure
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mahalanobis_map, cmap='viridis')
    plt.colorbar()
    plt.title("Mahalanobis Map")

    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("GT OOD Mask")
    plt.axis("off")

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
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

    parser.add_argument('--model',default="erfnet",choices=['erfnet', 'enet', 'bisenet'])
    parser.add_argument('--stats')
    parser.add_argument('--pca', type=str, default=None, help="Path to PCA result")
    parser.add_argument('--norm',default=None,choices=['minmax', 'z', 'gaussian',None])

    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:/semester 3/AML/project/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()


    if not os.path.exists('results4.txt'):
            open('results4.txt', 'w').close()
    file = open('results4.txt', 'a')

    if args.pca != None :
        args.pca = '../' + args.pca

    if args.stats != None :
        args.stats = '../' + args.stats



    path_list = [
        "D:/semester_3/AML/project/datasets/RoadAnomaly21/images/*.png",
        "D:/semester_3/AML/project/datasets/RoadObsticle21/images/*.webp",
        "D:/semester_3/AML/project/datasets/FS_LostFound_full/images/*.png",
        "D:/semester_3/AML/project/datasets/fs_static/images/*.jpg",
        "D:/semester_3/AML/project/datasets/RoadAnomaly/images/*.jpg"
    ]

    if args.model == "bisenet" :
        weightspath = '../trained_models/checkpoint20.pth'
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
    print ("✅Model and weights LOADED successfully")

    
    try :
        stats = torch.load(args.stats)
        mean = stats['mean']
        cov = stats['cov']
        print(f"✅ Stats loaded successfully")
    except :
        print(f"❌ Couln't load stats")


    if args.pca :
        try :
            pca = joblib.load(args.pca)
            print(f"✅ PCA loaded successfylly")
        except :
            print(f"❌ Couln't load PCA")
    else :
        pca = None


    model.eval()

    for current_path in path_list :

        ood_gts_list = []
        anomaly_score_list = []
        
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


        
        co_transform = MyCoTransform(enc=True, augment=False, height=512)
        resize_transform = transforms.Resize((512, 1024))

        for path in current_path :
            print(path)

            img = Image.open(path).convert("RGB")
            # img = resize_transform(img)
            # img_tensor, _ = co_transform(img, img)  # GT ignored
            img_tensor = input_transform(img)
            img_tensor = img_tensor.unsqueeze(0).to('cuda')


            with torch.no_grad():
                features = model(img_tensor, only_encode=True)


            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, 128]


            device = img_tensor.device
            mean  = mean.to(device)
            cov   = cov.to(device)

            mahalanobis_map = mahalanobis_score(features, mean, cov, pca_model=pca).view(H,W)

            # Optional: Upsample to original image size
            mahalanobis_map = mahalanobis_map.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 128]
            mahalanobis_map_up = F.interpolate(mahalanobis_map, size=(512, 1024), mode='bilinear', align_corners=False)
            mahalanobis_map_up = mahalanobis_map_up.squeeze().cpu().numpy()  # [512, 1024] 

            #min-max normalization suggestion to improve AUPRC
            if args.norm == 'minmax' :
                mahalanobis_map_up = (mahalanobis_map_up - mahalanobis_map_up.min()) / (mahalanobis_map_up.max() - mahalanobis_map_up.min() + 1e-8)

            #z-score normalization suggestion to improve AUPRC
            if args.norm == 'z' :
                mean = mahalanobis_map_up.mean()
                std = mahalanobis_map_up.std()
                mahalanobis_map_up = (mahalanobis_map_up - mean) / (std + 1e-8)

            # Gaussian smoothing
            if args.norm == 'gaussian' :
                mahalanobis_map_up = cv2.GaussianBlur(mahalanobis_map_up, ksize=(3, 3), sigmaX=1.5)            
            
    
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT:
                pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
                pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT:
                pathGT = pathGT.replace("jpg", "png")  

            mask = Image.open(pathGT)
            # mask_resized = mask.resize((1024,512), resample=Image.NEAREST)
            mask_resized = target_transform(mask)
            ood_gts = np.array(mask_resized)


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

            # visualize_anomaly_detection(img,mahalanobis_map_up,ood_gts)


            if 1 not in np.unique(ood_gts):
                continue              
            else:
                ood_gts_list.append(ood_gts)
                anomaly_score_list.append(mahalanobis_map_up)
            del features, mahalanobis_map_up, ood_gts, mask
            torch.cuda.empty_cache()


        file.write( "\n")

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)


        # print("ood_gts unique ",np.unique(ood_gts))

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)

        # print("Label unique values and counts:", np.unique(val_label, return_counts=True))


        if np.sum(val_label) == 0:
            print("No positive labels in validation set — skipping FPR@95")
        else:
            fpr = fpr_at_95_tpr(val_label, val_out)


        print(f'{dataset}     AUPRC score: {prc_auc*100.0}')
        print(f'{dataset}     FPR@TPR95: {fpr*100.0}')

        file.write((dataset + '     ' + 'AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()


if __name__ == '__main__':
    main()