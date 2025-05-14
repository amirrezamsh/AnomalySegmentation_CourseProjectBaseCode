
import torch

checkpoint = torch.load(r'C:\Users\ASUS\Desktop\ckpt-enet.pth', map_location='cpu')

for key in checkpoint:
    if 'weight' in key and ('fc' in key or 'classifier' in key):
        print(f"{key}: {checkpoint[key].shape}")
    print(key)

print(checkpoint['state_dict']['fullconv.weight'].shape)
# print(checkpoint['aux5_4.conv_out.1.weight'].shape)
# print(checkpoint['aux5_4.conv_out.1.bias'])

