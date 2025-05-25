import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
from argparse import ArgumentParser
from eval.transform import Relabel, ToLabel, Colorize
from eval.erfnet import ERFNet



# Cityscapes 20-class color map and names
CITYSCAPES_CLASSES = [
    ("unlabeled",      (0, 0, 0)),
    ("road",           (128, 64,128)),
    ("sidewalk",       (244, 35,232)),
    ("building",       (70, 70, 70)),
    ("wall",           (102,102,156)),
    ("fence",          (190,153,153)),
    ("pole",           (153,153,153)),
    ("traffic light",  (250,170, 30)),
    ("traffic sign",   (220,220,  0)),
    ("vegetation",     (107,142, 35)),
    ("terrain",        (152,251,152)),
    ("sky",            (70,130,180)),
    ("person",         (220, 20, 60)),
    ("rider",          (255,  0,  0)),
    ("car",            (0,  0,142)),
    ("truck",          (0,  0, 70)),
    ("bus",            (0, 60,100)),
    ("train",          (0, 80,100)),
    ("motorcycle",     (0,  0,230)),
    ("bicycle",        (119, 11, 32)),
]

def create_cityscapes_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for i, (_, color) in enumerate(CITYSCAPES_CLASSES):
        colormap[i] = color
    return colormap

def pred_to_color_image(pred):
    color_map = create_cityscapes_colormap()
    color_image = color_map[pred]
    return Image.fromarray(color_image)

def get_cityscapes_legend_patches():
    return [Patch(color=np.array(color)/255.0, label=label) for label, color in CITYSCAPES_CLASSES]

def main(args):

    enet_weightspath = "./trained_models/ENet" 
    bisenet_weightspath = "./trained_models/checkpoint20.pth" 
    erfnet_weightspath = "./trained_models/erfnet_pretrained.pth" 

    N_CLASSES = 20

    if args.model == 'enet':
        model = ENet(num_classes=N_CLASSES)
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'bisenet':
        model = BiSeNetV2(n_classes=N_CLASSES )
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'erfnet':
        model = ERFNet(N_CLASSES )
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
    

    if args.model == "enet":
        checkpoint = torch.load(enet_weightspath, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint['state_dict']
    if args.model == "bisenet":
        checkpoint = torch.load(bisenet_weightspath, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint['model_state_dict']
    if args.model == "erfnet":
        checkpoint = torch.load(erfnet_weightspath, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint
    
    
    model = load_my_state_dict(model, state_dict)
    model = model.eval().cuda()

    # ENet transformations

    def pil_to_long_tensor(pic):
        return torch.from_numpy(np.array(pic)).long()

    enet_img_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ])

    enet_lbl_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
        pil_to_long_tensor
    ])

    # BiseNet transformations
    
    mapping_20 = { 
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        7: 1, 8: 2,
        9: 0, 10: 0,
        11: 3, 12: 4, 13: 5,
        14: 0, 15: 0, 16: 0,
        17: 6, 18: 0, 19: 7, 20: 8, 21: 9,
        22: 10, 23: 11, 24: 12, 25: 13, 26: 14,
        27: 15, 28: 16,
        29: 0, 30: 0,
        31: 17, 32: 18, 33: 19,
        255:0
    }

    # Create fast lookup table (for performance)
    mapping_array = np.zeros(256, dtype=np.uint8)
    for k, v in mapping_20.items():
        mapping_array[k] = v

    # Transformation function for the label
    def pil_to_mapped_tensor(pic):
        label = np.array(pic)  # Convert PIL to numpy
        # label[label == -1] = 255
        label = mapping_array[label]  # Apply mapping
        return torch.from_numpy(label).long()


    bisenet_img_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


    bisenet_lbl_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
        pil_to_mapped_tensor,
    ])

    erfnet_img_transform = Compose([
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ])
    erfnet_lbl_transform = Compose([
        Resize(512, Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),   #ignore label to 19
    ])


    img_path  = "../datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_003357_leftImg8bit.png"
    gt_path   = "../datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_003357_gtFine_labelIds.png"
    # img_path  = "../datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001236_leftImg8bit.png"
    # gt_path   = "../datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_001236_gtFine_labelIds.png"
    # img_path = r"D:\semester_3\AML\project\datasets\RoadAnomaly21\images\1.png" 
    # gt_path = r"D:\semester_3\AML\project\datasets\RoadAnomaly21\labels_masks\1.png" 
    
    img       = Image.open(img_path)
    gt_label  = Image.open(gt_path)

    if args.model == "enet":
        x = enet_img_transform(img).unsqueeze(0).cuda()
        y = enet_lbl_transform(gt_label).unsqueeze(0).cuda()
    elif args.model == 'bisenet':
        x = bisenet_img_transform(img).unsqueeze(0).cuda()
        y = bisenet_lbl_transform(gt_label).unsqueeze(0).cuda()
    elif args.model == 'erfnet':
        x = enet_img_transform(img).unsqueeze(0).cuda()
        y = enet_lbl_transform(gt_label).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.argmax(1).squeeze(0).cpu().numpy()

        print("logits shape ", logits.shape)


    # Convert prediction to RGB image

    if args.model == "erfnet" :
        pred = pred + 1

    print("Input shape: ", x.shape)
    print("Logits shape:", logits.shape)
    print("Unique preds:", np.unique(pred))

    pred_color = pred_to_color_image(pred)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((1024, 512)))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_color)
    plt.title("Prediction with Color Map")
    plt.axis("off")

    # Add legend
    patches = get_cityscapes_legend_patches()
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--state')
    parser.add_argument('--model', default="enet")
    main(parser.parse_args())
