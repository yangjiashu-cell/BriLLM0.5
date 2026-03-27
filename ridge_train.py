from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler  # 新增：标准化
from scipy.stats import pearsonr  # Add for Pearsonr metric (LEA-inspired)
import numpy as np
from PIL import Image
import torch.nn as nn
from open_clip import create_model_and_transforms
from safetensors.torch import load_file
import torch
from tqdm import tqdm
import pickle
import logging

logging.basicConfig(filename='ridge_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting Ridge alignment on ds001246 real data (improved version)')

# Load cond_proj and model
cond_proj = nn.Linear(1280, 32).cuda()
#cond_proj.load_state_dict(torch.load('cond_proj.pth'))
state = torch.load('cond_proj.pth')
cond_proj.load_state_dict(state['cond_proj'])
cond_proj.eval()


model, _, preprocess = create_model_and_transforms('ViT-bigG-14', pretrained=None)
state_dict = load_file('/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', device='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.visual.cuda().eval()

# Load data（现在应该是只有真实图像的2401个）
fMRI_cls = np.load('/root/BriLLM0.5/ds001246_fMRI_cls.npy')  # (N, 768) [CLS] from MAE
with open('/root/BriLLM0.5/ds001246_images_pil.pkl', 'rb') as f:
    images_pil = pickle.load(f)

print(f'Loaded {fMRI_cls.shape[0]} trials, using {len(images_pil)} images')

# Batch extract CLIP conditions
batch_size = 32
image_conds = []

with torch.no_grad():
    for i in tqdm(range(0, len(images_pil), batch_size), desc='Extracting CLIP conditions'):
        batch_imgs = images_pil[i:i+batch_size]
        inputs = []
        for img in batch_imgs:
            try:
                inputs.append(preprocess(img))
            except Exception as e:
                logging.warning(f'Error preprocessing image {i}: {e}, using black dummy')
                dummy = Image.new('RGB', (224, 224), (0, 0, 0))
                inputs.append(preprocess(dummy))
        inputs = torch.stack(inputs).to('cuda')

        vision_out = model(inputs)
        if vision_out.dim() > 2:
            vision_out = vision_out.mean(dim=1)  # 确保是 (batch, 1280)
        cond = cond_proj(vision_out).cpu().numpy()  # (batch, 32)
        image_conds.append(cond)

image_conds = np.concatenate(image_conds, axis=0)  # (N, 32)

# === 关键修改：数据标准化（非常重要！）===
# === 关键修改：数据标准化（非常重要！）===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

fMRI_cls_scaled = scaler_X.fit_transform(fMRI_cls)          # (N, 768) 零均值单位方差
image_conds_scaled = scaler_y.fit_transform(image_conds)  # (N, 32)
import pickle
pickle.dump(scaler_X, open('scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open('scaler_y.pkl', 'wb'))

# === 扩大 alpha 搜索范围（对标准化数据更敏感），加 LEA alpha=500 ===
alphas = np.logspace(-15, 0, 100)  # 从 1e-3 到 1e3，20个点
alphas = np.unique(np.append(alphas, 5e-9))  # 加 LEA Page5 alpha=500

ridge = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)  # 加 fit_intercept=True 允许偏置，5折 CV
ridge.fit(fMRI_cls_scaled, image_conds_scaled)

# 保存系数（注意：保存前需反标准化回原始尺度？通常不需要，因为eval时也会标准化）
# 但为了eval一致性，这里保存原始尺度系数（对未标准化数据有效）
from sklearn.linear_model import Ridge
ridge_raw = Ridge(alpha=ridge.alpha_, fit_intercept=True)  # 用最佳alpha在原始数据上重新fit（使用Ridge而非RidgeCV）
ridge_raw.fit(fMRI_cls, image_conds)
np.save('ridge_coef.npy', ridge_raw.coef_)  # 保存用于eval的系数（未标准化）
# 计算原始数据上的R2（更直观）
r2_raw = ridge_raw.score(fMRI_cls, image_conds)

# 加 Pearsonr metric (LEA Sec3, avg ~15.86 for GOD)
pearson_scores = []
for i in range(image_conds.shape[1]):
    pred = ridge_raw.predict(fMRI_cls)[:, i]
    pearson_scores.append(pearsonr(pred, image_conds[:, i])[0])
avg_pearson = np.mean(pearson_scores)

print(f'Best alpha (on scaled): {ridge.alpha_:.2e}')
print(f'R2 score (on raw data): {r2_raw:.4f}')
print(f'Avg Pearsonr (on raw data): {avg_pearson:.4f}')
logging.info(f'Ridge complete: best_alpha_scaled={ridge.alpha_:.2e}, R2_raw={r2_raw:.4f}, avg_pearson={avg_pearson:.4f}')