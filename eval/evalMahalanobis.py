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

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import torch.nn.functional as F
from train.main import MyCoTransform
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF



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
    parser.add_argument(
    "--input",
    default="D:/semester_3/AML/project/datasets/RoadObsticle21/images/*.webp",
    help="Glob pattern to match images"
)
    # parser.add_argument('--method', default='msp', choices=['msp', 'maxlogit', 'entropy'],
    #                 help="Anomaly scoring method: msp, maxlogit, or entropy")
    
    parser.add_argument('--model',default="erfnet",choices=['erfnet', 'enet', 'bisenet'])
    
    # parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits (used in MSP)')


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

    mean_vector = torch.load('../mean_vector.pt').to('cuda')      # shape: [128]
    cov = torch.load('../covariance_matrix.pt')
    # Regularization (apply again to be safe)
    reg_lambda = 1e-6
    identity = torch.eye(cov.shape[0], device=cov.device)
    cov += reg_lambda * identity
    # Compute inverse
    inv_cov = torch.inverse(cov)

    
    co_transform = MyCoTransform(enc=True, augment=False, height=512)
    resize_transform = transforms.Resize((512, 1024))

    

    for path in args.input :
        print(path)
        # images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        # images = images.permute(0,3,1,2)
        # print("images shape ",images.shape)
        img = Image.open(path).convert("RGB")
        img = resize_transform(img)
        img_tensor, _ = co_transform(img, img)  # GT ignored
        img_tensor = img_tensor.unsqueeze(0).to('cuda')

        # print("img_tensor shape ",img_tensor.shape) #([1, 3, 720, 1280])

        with torch.no_grad():
            features = model(img_tensor, only_encode=True)

        # print("features shape1 ",features.shape) #([1, 128, 90, 160])
            

        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, 128]
        # features = features.squeeze(0)  # [128, H, W]
        # features = features.permute(1, 2, 0).reshape(-1, 128)  # [H*W, 128]

        # print("features shape ",features.shape) #([14400, 128])


        # features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6) #wrong


        print("Feature mean:", features.mean().item())
        print("Feature std:", features.std().item())
        print("Feature max:", features.max().item())
        print("Feature min:", features.min().item())


        # Subtract mean
        delta = features - mean_vector

        # Mahalanobis distance: sqrt((x - μ)^T Σ⁻¹ (x - μ))
        # Equivalent to: sqrt(sum((delta @ inv_cov) * delta, dim=1))
        inv_cov = inv_cov.to(delta.device)

        # print("Delta contains NaN:", torch.isnan(delta).any().item())
        # print("inv_cov contains NaN:", torch.isnan(inv_cov).any().item())
        # print("Delta contains Inf:", torch.isinf(delta).any().item())
        # print("inv_cov contains Inf:", torch.isinf(inv_cov).any().item())


        # mahalanobis = torch.sqrt((delta @ inv_cov) * delta).sum(dim=1)  # [H*W] #this gives nan in result
        mahalanobis = torch.sqrt(torch.einsum('bi,ij,bj->b', delta, inv_cov, delta))

        # mahalanobis = (mahalanobis - mahalanobis.min()) / (mahalanobis.max() - mahalanobis.min()) #comment this out, I get better results without this 

        # print("Min Mahalanobis distance:", mahalanobis.min().item())
        # print("Max Mahalanobis distance:", mahalanobis.max().item())

        # Reshape back to image size
        mahalanobis_map = mahalanobis.view(H, W)  # [64, 128]

        # Optional: Upsample to original image size
        mahalanobis_map = mahalanobis_map.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 128]
        mahalanobis_map_up = F.interpolate(mahalanobis_map, size=(512, 1024), mode='bilinear', align_corners=False)
        mahalanobis_map_up = mahalanobis_map_up.squeeze().cpu().numpy()  # [512, 1024] 

        #min-max normalization suggestion to improve AUPRC
        # mahalanobis_map_up = (mahalanobis_map_up - mahalanobis_map_up.min()) / (mahalanobis_map_up.max() - mahalanobis_map_up.min() + 1e-8)


        #z-score normalization suggestion to improve AUPRC
        # mean = mahalanobis_map_up.mean()
        # std = mahalanobis_map_up.std()
        # mahalanobis_map_up = (mahalanobis_map_up - mean) / (std + 1e-8)


        # Gaussian smoothing
        mahalanobis_map_up = cv2.GaussianBlur(mahalanobis_map_up, ksize=(3, 3), sigmaX=1.5)


        print("mahalanobismapup shape ",mahalanobis_map_up.shape)
        
        
   
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask_resized = mask.resize((1024,512), resample=Image.NEAREST)
        ood_gts = np.array(mask_resized)

        print("oodgts shape ",ood_gts.shape)

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

    print("ood_gts unique ",np.unique(ood_gts))

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print("val_label shape:", val_label.shape)
    # print("val_label values:", val_label.flatten()[:10])

    print("val_out shape:", val_out.shape)
    # print("val_out values:", val_out.flatten()[:10])


    prc_auc = average_precision_score(val_label, val_out)

    print("Label unique values and counts:", np.unique(val_label, return_counts=True))


    if np.sum(val_label) == 0:
        print("No positive labels in validation set — skipping FPR@95")
    else:
        fpr = fpr_at_95_tpr(val_label, val_out)


    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    # file.write((dataset + '     ' + args.method + '     ' + str(args.temperature) + '    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()


if __name__ == '__main__':
    main()