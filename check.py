import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
import transforms as ext_transforms
from argparse import ArgumentParser

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
    # bisenet_weightspath = "./trained_models/model_final_v2_city.pth" 

    bisenet_weightspath = "./trained_models/checkpoint15.pth" 

    N_CLASSES = 20

    if args.model == 'enet':
        model = ENet(num_classes=N_CLASSES)
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'bisenet':
        model = BiSeNetV2(n_classes=N_CLASSES)
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
        state_dict = torch.load(enet_weightspath, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = state_dict['state_dict']
    if args.model == "bisenet":
        state_dict = torch.load(bisenet_weightspath, map_location=lambda storage, loc: storage, weights_only=False)

    model = load_my_state_dict(model, state_dict['model_state_dict'])
    model = model.eval().cuda()

    enet_img_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ])

    enet_lbl_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
        ext_transforms.PILToLongTensor(),
    ])

    bisenet_img_transform = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    def pil_to_long_tensor(pic):
        return torch.from_numpy(np.array(pic)).long()

    bisenet_lbl_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
        pil_to_long_tensor,
    ])

    img_path  = "../datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_003357_leftImg8bit.png"
    gt_path   = "../datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_003357_gtFine_labelIds.png"
    # img_path  = "../datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001236_leftImg8bit.png"
    # gt_path   = "../datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_001236_gtFine_labelIds.png"
    img       = Image.open(img_path)
    gt_label  = Image.open(gt_path)

    if args.model == "enet":
        x = enet_img_transform(img).unsqueeze(0).cuda()
        y = enet_lbl_transform(gt_label).unsqueeze(0).cuda()
    elif args.model == 'bisenet':
        x = bisenet_img_transform(img).unsqueeze(0).cuda()
        y = bisenet_lbl_transform(gt_label).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.argmax(1).squeeze(0).cpu().numpy()

    

    # Convert prediction to RGB image

    if args.model == "bisenet" :
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
