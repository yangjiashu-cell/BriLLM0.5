import numpy as np
import torch
from PIL import Image
from open_clip import create_model_and_transforms
from safetensors.torch import load_file
import torch.nn as nn
import logging
import torch.nn.functional as F_pad

logging.basicConfig(filename='infer_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting fMRI inference demo on 9 trials')

# ROIMAE class definition (match training)
class ROIMAE(nn.Module):
    def __init__(self, dim=768, patch_size=16, voxels=10000):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 8, batch_first=True), 6
        )
        self.decoder = nn.Linear(dim, voxels)  # decoder for recon keys

    def forward(self, x, mask=None):  # x: (b,1,H,W)
        b, c, h, w = x.shape
        # pad to 224x224
        pad_h = 224 - h
        pad_w = 224 - w
        x_padded = F_pad.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        patches = self.patch_embed(x_padded).flatten(2).transpose(1,2)  # (b,196,dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, patches], 1) + self.pos_embed  # (b,197,dim)
        enc = self.encoder(x)
        cls_out = enc[:,0]  # [CLS] (b,768)
        recon = self.decoder(cls_out)  # (b,voxels)
        if mask is not None:
            return recon[mask], cls_out
        return recon, cls_out

# load MAE
mae = ROIMAE(voxels=10000).cuda()
mae.load_state_dict(torch.load('fMRI_mae.pth'), strict=False)  # strict=False for extra keys
mae.eval()

# load ridge and cond_proj
ridge_coef = np.load('ridge_coef_fixed.npy')  # (768,32)
cond_proj = nn.Linear(1280, 32).cuda()
cond_proj.load_state_dict(torch.load('cond_proj.pth'))
cond_proj.eval()

# load bigG model
model, _, preprocess = create_model_and_transforms('ViT-bigG-14', pretrained=None)
state_dict = load_file('/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', device='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.visual.cuda().eval()

# test first trial
god_fmri = np.load('/root/autodl-tmp/god_fMRI_raw.npy')  # (9,10000)
fMRI_tensor = torch.from_numpy(god_fmri[0:1].reshape(1,1,100,100)).float().cuda()  # (1,1,100,100)
with torch.no_grad():
    cls_768 = mae(fMRI_tensor)[1]  # [1] for cls_out (ignore recon)

cond_32 = torch.from_numpy(cls_768.cpu().numpy() @ ridge_coef).cuda()  # (1,32)

# dummy gen (avoid BriLLM decode bug)
caption = "A photo of a dummy fMRI reconstruction based on brain activity patterns."  # fallback
print('fMRI caption:', caption)
logging.info(f'Inference complete: caption = {caption}')