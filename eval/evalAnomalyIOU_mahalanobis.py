
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
from train.main import MyCoTransform
from torchvision import transforms
import joblib

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import torch.nn.functional as F
import matplotlib.pyplot as plt


def mahalanobis_score(features, mean, inv_cov, pca_model=None):

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
    if not isinstance(inv_cov, torch.Tensor):
        inv_cov = torch.tensor(inv_cov, dtype=torch.float32)

    # Move mean and cov to the same device as features
    device = mean.device
    features = features.to(device)

    # Compute Mahalanobis score
    delta = features - mean
    scores = torch.einsum('nd,dd,nd->n', delta, inv_cov, delta)
    return scores


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
    
    parser.add_argument('--model',default="erfnet",choices=['erfnet', 'enet', 'bisenet'])    
    
    parser.add_argument('--invcov',default='../inv_cov_reduced_64d.npy')
    parser.add_argument('--mean',default='../mean_reduced_64d.npy')
    parser.add_argument('--pca', type=str, default=None, help="Path to PCA result")

    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=r"D:/semester 3/AML/project/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.pca != None :
        args.pca = '../' + args.pca



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

    try :
      inv_cov = np.load(args.invcov)
      mean = np.load(args.mean)
      print(f"✅ Mean and inv_cov loaded successfully")
    except :
      print(f"❌ Couln't load invcov or mean")

    inv_cov = torch.from_numpy(inv_cov)
    mean = torch.from_numpy(mean)

    if args.pca :
        try :
            pca = joblib.load(args.pca)
            print(f"✅ PCA loaded successfylly")
        except :
            print(f"❌ Couln't load PCA")
    else :
        pca = None

    co_transform = MyCoTransform(enc=True, augment=False, height=512)
    resize_transform = transforms.Resize((512, 1024)) 


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
            img = Image.open(path).convert("RGB")
            img = resize_transform(img)
            img_tensor, _ = co_transform(img, img)  # GT ignored
            # img_tensor = input_transform(img)

            # transform = transforms.ToTensor()  # Converts to [C, H, W] and scales to [0, 1]
            # img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0).to('cuda')

            with torch.no_grad():
                features = model(img_tensor, only_encode=True)

            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, 128]
            device = img_tensor.device
            mean  = mean.to(device)
            inv_cov   = inv_cov.to(device)

            mahalanobis_map = mahalanobis_score(features, mean, inv_cov, pca_model=pca).view(H,W)

            # Optional: Upsample to original image size
            mahalanobis_map = mahalanobis_map.unsqueeze(0).unsqueeze(0)  
            # mahalanobis_map_up = F.interpolate(mahalanobis_map, size=(720, 1280), mode='bilinear', align_corners=False)
            # mahalanobis_map_up = mahalanobis_map_up.squeeze().cpu().numpy()
            # anomaly_result = mahalanobis_map_up

            # threshold = np.percentile(anomaly_result, PERCENTILE)
            # binary_map = (anomaly_result > threshold).astype(np.uint8)

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

            
            ood_gts = torch.from_numpy(ood_gts).unsqueeze(0).unsqueeze(0).to(device).long()


            ood_gts_np = ood_gts.cpu().numpy()

            gt_H, gt_W = ood_gts_np.shape[2], ood_gts_np.shape[3]

            mahalanobis_map_up = F.interpolate(mahalanobis_map, size=(gt_H, gt_W), mode='bilinear', align_corners=False)
            mahalanobis_map_up = mahalanobis_map_up.squeeze().cpu().numpy()
            anomaly_result = mahalanobis_map_up

            threshold = np.percentile(anomaly_result, PERCENTILE)
            binary_map = (anomaly_result > threshold).astype(np.uint8)
            binary_map = torch.from_numpy(binary_map).unsqueeze(0).unsqueeze(0).to(device).long()




            if 1 not in np.unique(ood_gts_np):
                continue              
            else:
              iouEvalVal_overall.addBatch(binary_map, ood_gts)
              iouEvalVal_dataset.addBatch(binary_map, ood_gts)
            del features, anomaly_result, ood_gts, mask
            torch.cuda.empty_cache()      

        iouVal_dataset, iou_classes_dataset = iouEvalVal_dataset.getIoU()
        iouStr = getColorEntry(iouVal_dataset)+'{:0.2f}'.format(iouVal_dataset*100) + '\033[0m'
        print (f"MEAN IoU for dataset {dataset}: {iouStr}% ")
    iouVal, iou_classes = iouEvalVal_overall.getIoU()


    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print (f"Overall MEAN IoU based on mahalanobis : {iouStr}%")


if __name__ == '__main__':
    main()