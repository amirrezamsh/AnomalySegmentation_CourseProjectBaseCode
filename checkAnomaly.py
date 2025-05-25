import sys
import os
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from argparse import ArgumentParser
from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize




NUM_CLASSES = 20

bisenet_img_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])



def main():
    parser = ArgumentParser()
    
    parser.add_argument('--method', default='msp', choices=['msp', 'maxlogit', 'entropy'],
                    help="Anomaly scoring method: msp, maxlogit, or entropy")
    
    parser.add_argument('--model',default="erfnet",choices=['erfnet', 'enet', 'bisenet'])

    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits (used in MSP)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.model == "bisenet" :
        weightspath = './trained_models/checkpoint20.pth'
    elif args.model == "erfnet" :
        weightspath = './trained_models/erfnet_pretrained.pth'
    elif args.model == "enet" :
        weightspath = '../save/ENet_Cityscapes/ENet'

    print ("Loading weights: " + weightspath)

    if args.model == "erfnet" :
        print("model is erfnet")
        # model = ERFNet(NUM_CLASSES)
    elif args.model == "enet" :
        print("model is enet")
        model = ENet(NUM_CLASSES)
    elif args.model == "bisenet" :
        print("model is bisenet")
        model = BiSeNetV2(NUM_CLASSES)

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

    model = model.eval().to(device)

    img_path  = r"D:\semester_3\AML\project\datasets\RoadAnomaly21\images\7.png"
    
    image = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))).unsqueeze(0).float()
    image = image.permute(0,3,1,2)
    image = image.to(device)

        

    # 
    # image = Image.open(img_path).convert("RGB")  # Ensure it's in RGB mode

    # # Apply the transform
    # transformed_image = bisenet_img_transform(image)

    # # If your model expects a batch, add a batch dimension
    # transformed_image = transformed_image.unsqueeze(0)
    # transformed_image = transformed_image.to(device)

    # 

    with torch.no_grad():
        result = model(image)
        
        if isinstance(result, tuple):
            result = result[0]

        logits = result.squeeze(0).data.cpu().numpy()

        if args.method == "msp" :
          temperature = getattr(args, 'temperature', 1.0)

          softmax_probs = F.softmax(result / temperature, dim=1)  # Shape: [B, C, H, W]
          msp, predicted = torch.max(softmax_probs, dim=1)  # Shape: [B, H, W]
          anomaly_result = msp
          anomaly_result = anomaly_result.squeeze(0).data.cpu().numpy()
          
          print("anomaly result shape ",anomaly_result.shape)


        elif args.method == "maxlogit" :
            # MaxLogit anomaly score
            anomaly_result = -np.max(logits, axis=0)  # shape: (H, W)

        elif args.method == "entropy" :

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

    
    image_tensor = image[0]  # Remove batch dimension → shape [3, 512, 1024]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # → shape [512, 1024, 3]
    image_np = (image_np * 255).astype(np.uint8)  # Convert from [0, 1] float to uint8

    # Normalize anomaly_result to [0, 255]
    anomaly_map = anomaly_result.astype(np.float32)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)

    # Apply a heatmap using OpenCV
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    # Display the images
    plt.figure(figsize=(15, 5))

    # 1. Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_np)
    plt.axis("off")

    # 2. Anomaly Heatmap
    plt.subplot(1, 3, 2)
    plt.title("Anomaly Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    # 3. Overlay Image
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

      


if __name__ == '__main__':
    main()


          







      