from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split  # 新增
import numpy as np
from PIL import Image
import torch.nn as nn
from open_clip import create_model_and_transforms
from safetensors.torch import load_file
import torch
from tqdm import tqdm
import logging
import pickle  # 新增 for pickle load

logging.basicConfig(filename='ridge_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting Ridge alignment on real trials')

# load cond_proj and model (不变)
cond_proj = nn.Linear(1280, 32).cuda()
cond_proj.load_state_dict(torch.load('cond_proj.pth'))
cond_proj.eval()

model, _, preprocess = create_model_and_transforms('ViT-bigG-14', pretrained=None)
state_dict = load_file('/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', device='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.visual.cuda().eval()

# 修改：用preprocess save的real fMRI [CLS] (假设你已跑MAE on god_fMRI_raw_real.npy得cls)
# 注意: 如果fMRI_cls是MAE [CLS], 先跑MAE预训get new cls.npy; 这里假设已有或用raw voxels代 (需fix)
fMRI_cls = np.load('/root/BriLLM0.5/god_FMRI_raw_real.npy')  # (45,10000) raw voxels; 如果是cls, 改'god_fMRI_cls_real.npy'

# 修改：用real images from pickle (preprocess save)
with open('/root/BriLLM0.5/god_images_list.pkl', 'rb') as f:
    god_images = pickle.load(f)

num_trials = len(god_images)
logging.info(f'Loaded {num_trials} real trials')
print(f'Loaded {num_trials} real trials')
if num_trials != len(fMRI_cls):
    logging.error('fMRI and images mismatch!')
    raise ValueError('fMRI and images mismatch!')

image_conds = []
for i in tqdm(range(num_trials), desc='Ridge alignment'):
    img = god_images[i].resize((224, 224))
    inputs = preprocess(img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        vision_out = model(inputs)  # (1,1280)
        vision_out = vision_out.mean(1) if vision_out.dim() > 2 else vision_out
        cond = cond_proj(vision_out).cpu().numpy()[0]
    image_conds.append(cond)
    logging.info(f'Trial {i}: cond std = {np.std(cond):.4f}')
image_conds = np.array(image_conds)  # (num_trials,32)

# 新增：debug Y diversity
print('Unique cond rows:', np.unique(image_conds, axis=0).shape)  # 应~ (num_trials,32)
y_std = np.std(image_conds, axis=0).mean()
print('Y std per dim:', y_std)  # >0.1 好
if y_std < 0.01:
    logging.warning('Degenerate Y: Use diverse real images!')

# 新增：OOF split (80/20)，fit on train, score on test
X_train, X_test, y_train, y_test = train_test_split(fMRI_cls, image_conds, test_size=0.2, random_state=42)
ridge = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100,500])
ridge.fit(X_train, y_train)
train_r2 = ridge.score(X_train, y_train)
test_r2 = ridge.score(X_test, y_test)
print(f'Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')  # Test ~0.2-0.4 预期
np.save('ridge_coef_fixed.npy', ridge.coef_)  # 只保存好coef
print("ridge_coef 维度:", ridge.coef_.shape)
logging.info(f'Ridge complete: alpha={ridge.alpha_}, Train R2={train_r2}, Test R2={test_r2}')