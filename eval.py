import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from safetensors.torch import load_file
import torch
import torch.nn as nn
import logging
import json
import pickle

logging.basicConfig(filename='eval_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting evaluation on ds001246 full data (2401 trials)')

cond_proj = nn.Linear(1280, 32).cuda()
state = torch.load('cond_proj.pth')
cond_proj.load_state_dict(state['cond_proj'])
#cond_proj.load_state_dict(torch.load('cond_proj.pth'))
cond_proj.eval()

# Load bigG model (不变)
bigg_model, _, bigg_preprocess = create_model_and_transforms('ViT-bigG-14', pretrained=None)
state_dict = load_file('/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', device='cpu')
bigg_model.load_state_dict(state_dict, strict=False)
bigg_model = bigg_model.visual.cuda().eval()

# Load CLIP ViT-B-16 for evaluation
clip_model, _, clip_preprocess = create_model_and_transforms('ViT-B-16', pretrained=None)
clip_state_dict = load_file('/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_model.safetensors', device='cpu')
clip_model.load_state_dict(clip_state_dict, strict=False)
clip_model.cuda().eval()
clip_tokenizer = get_tokenizer('ViT-B-16')

# Load Ridge (注意：ridge_coef.npy 是 (32, 768)，需转置为 (768,32) 以匹配 (1,768) @ (768,32) -> (1,32))
ridge_coef = np.load('ridge_coef.npy').T  # 转置到 (768,32)

from test_condition import gen

# Load data (与预处理一致，使用2401个trial)
fMRI_cls = np.load('/root/BriLLM0.5/ds001246_fMRI_cls.npy')  # (2401, 768) [CLS]
with open('/root/BriLLM0.5/ds001246_images_pil.pkl', 'rb') as f:
    real_images = pickle.load(f)  # list of PIL Images, length=2401

print(f'Loaded {fMRI_cls.shape[0]} trials for evaluation')

clips = []
captions = []

with torch.no_grad():
    for i in tqdm(range(len(fMRI_cls)), desc='Evaluation on full ds001246'):
        cls_768 = torch.from_numpy(fMRI_cls[i:i+1].astype(np.float32)).cuda()  # (1,768)
        
        logging.info(f'Trial {i}: cls_768 shape={cls_768.shape}, ridge_coef shape={ridge_coef.shape}')
        
        cond_np = cls_768.cpu().numpy() @ ridge_coef  # (1,768) @ (768,32) -> (1,32)
        cond = torch.from_numpy(cond_np.astype(np.float32)).cuda()  # (1,32)

        caption = gen(cond=cond, temperature=0.5, max_new_tokens=20)
        captions.append(caption)

        gt_img = real_images[i]
        try:
            gt_input = clip_preprocess(gt_img).unsqueeze(0).cuda()
        except Exception:
            # 如果图像损坏，用black dummy避免crash
            dummy = Image.new('RGB', (224, 224), (0, 0, 0))
            gt_input = clip_preprocess(dummy).unsqueeze(0).cuda()

        gt_emb = clip_model.encode_image(gt_input)
        text_input = clip_tokenizer([caption]).cuda()
        cap_emb = clip_model.encode_text(text_input)

        sim = F.cosine_similarity(gt_emb, cap_emb).item()
        clips.append(sim)
        logging.info(f'Trial {i}: caption="{caption}", CLIP_score={sim:.4f}')

avg_clip = np.mean(clips)
print(f'Avg CLIP score on {len(clips)} trials: {avg_clip:.4f}')

results = {
    'clips': clips,
    'avg': float(avg_clip),
    'trials': len(clips),
    'captions': captions
}
with open('eval_results.json', 'w') as f:
    json.dump(results, f)

logging.info(f'Evaluation complete, Avg CLIP: {avg_clip:.4f}')