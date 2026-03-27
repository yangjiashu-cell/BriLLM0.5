import nibabel as nib
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import logging
import pickle

logging.basicConfig(filename='/root/BriLLM0.5/preprocess_ds001246_log.txt', 
                    level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting ds001246 preprocess with fixes: use raw bold, image_file column, basename handling, ImageNet only')

base_dir = '/root/autodl-tmp/ds001246/derivatives/preproc-spm/output'
stim_base = '/root/autodl-tmp/ds001246/BOLD5000_Stimuli/Scene_Stimuli/Original_Images'
sub_dirs = ['ImageNet']  # ds001246 (GOD) 只用 ImageNet 刺激，你已解压到这里

# 使用原始 raw bold 文件（ds001246 标准下载没有 preproc derivatives）
bold_files = sorted(glob(f'{base_dir}/sub-*/ses-perception*/func/*_task-perception_run-*_bold_preproc.nii.gz'))
if len(bold_files) == 0:
    logging.error('No raw bold files found! Check dataset download and path.')
    print('Error: No bold files found. Confirm ds001246 is fully downloaded.')
    exit(1)

# Derive events path from raw bold
def get_events_file(bold_file):
    basename = os.path.basename(bold_file)
    events_filename = basename.replace('_bold.nii.gz', '_events.tsv')
    parts = basename.split('_')
    sub = parts[0]
    ses = parts[1]
    orig_dir = os.path.join(base_dir, sub, ses, 'func')
    return os.path.join(orig_dir, events_filename)

events_files = [get_events_file(f) for f in bold_files]

logging.info(f'Found {len(bold_files)} raw bold files (expected ~76 for 4 subjects)')

fmri_list = []
image_list = []
missing_count = 0
total_trials = 0

for bold_file, events_file in tqdm(zip(bold_files, events_files), total=len(bold_files)):
    if not os.path.exists(events_file):
        logging.warning(f'No events file for {bold_file}: {events_file}')
        continue
    
    try:
        img = nib.load(bold_file)
        data = img.get_fdata()  # (x, y, z, time)
        TR = round(img.header.get_zooms()[-1], 1)  # GOD TR=3.0s
        if TR != 3.0:
            logging.warning(f'Unexpected TR {TR} in {bold_file}')
        
        events = pd.read_csv(events_file, sep='\t')
        
        # ds001246 (GOD) 使用 image_file 列，没有 event_type，所有行都是 stimulus trial
        if 'image_file' not in events.columns:
            logging.warning(f'No image_file column in {events_file}, columns: {list(events.columns)}')
            continue
        
        for _, row in events.iterrows():
            total_trials += 1
            onset = row['onset']
            
            if not np.isfinite(onset):
                logging.warning(f'Skipping non-finite onset: {onset}')
                continue
            
            # image_file 格式如 n03626115_19498.JPEG 或带路径，取 basename
            full_stim = str(row['image_file']).strip()
            if not full_stim or full_stim in ['n/a', 'nan']:
                continue
            
            stim_name = os.path.basename(full_stim)  # 安全取文件名
            base_name, ext = os.path.splitext(stim_name)  # 分离扩展名
            
            # 尝试常见大小写变体
            possible_exts = ['.JPEG', '.jpg', '.JPG', '.jpeg']
            if ext:  # 如果已有扩展名，也尝试原样
                possible_exts = [ext] + possible_exts
            
            img_path = None
            for sub in sub_dirs:
                for e in possible_exts:
                    candidate = os.path.join(stim_base, sub, base_name + e)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break
            
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
            else:
                pil_img = Image.new('RGB', (224, 224), (255, 0, 0))  # 红色 dummy
                missing_count += 1
                logging.warning(f'Missing image: {stim_name} (tried in {sub_dirs})')
            
            # fMRI extract: +4s ~ +10s average
            start_tr = int((onset + 4) // TR)
            end_tr = int((onset + 10) // TR)
            if start_tr >= data.shape[-1] or end_tr > data.shape[-1] or start_tr >= end_tr:
                logging.warning(f'Invalid TR range: start={start_tr}, end={end_tr}, vols={data.shape[-1]}')
                continue
            
            vol = data[..., start_tr:end_tr].mean(axis=-1)
            voxels = vol.flatten()[:10000]  # 前 10000 voxels (与你之前一致)
            
            fmri_list.append(voxels)
            image_list.append(pil_img)
    
    except Exception as e:
        logging.error(f'Error processing {bold_file}: {str(e)}')
        continue

fmri_array = np.array(fmri_list)  # (N, 10000)
logging.info(f'Processed {len(fmri_array)} valid trials, {missing_count} missing images out of {total_trials}')

np.save('/root/BriLLM0.5/ds001246_fmri_10k.npy', fmri_array)
with open('/root/BriLLM0.5/ds001246_images_pil.pkl', 'wb') as f:
    pickle.dump(image_list, f)
np.save('/root/BriLLM0.5/ds001246_images_np.npy', np.array([np.array(img) for img in image_list]))

print(f'完成！得到 {len(fmri_array)} 个有效 fMRI-image 配对')
print(f'缺失图像数: {missing_count} / {total_trials} （如果缺失很多，检查 ImageNet 目录下文件名是否匹配 .JPEG/.jpg 大小写）')
logging.info('Preprocess finished successfully')