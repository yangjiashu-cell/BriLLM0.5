import nibabel as nib
import numpy as np
from glob import glob
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import os
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

base_dir = '/root/autodl-tmp/ds001246'
deriv_dir = os.path.join(base_dir, 'derivatives/preproc-spm/output')  # Preprocessed bold path
stim_base = '/root/autodl-tmp/ds001246/BOLD5000_Stimuli/Scene_Stimuli/Original_Images'  # 刺激根目录
sub_dirs = ['COCO', 'ImageNet', 'Scene']  # 三个子目录

# Find all preprocessed bold files
bold_files = sorted(glob(f'{deriv_dir}/sub-*/ses-perception*/func/*_task-perception_run-*_bold_preproc.nii.gz'))

def get_events_file(bold_file):
    parts = os.path.basename(bold_file).split('_')
    sub = parts[0]
    ses = parts[1]
    task = parts[2]
    run = parts[3].split('.')[0]  # 防止有扩展名干扰
    orig_dir = os.path.join(base_dir, sub, ses, 'func')
    events_filename = f'{sub}_{ses}_{task}_{run}_events.tsv'
    return os.path.join(orig_dir, events_filename)

events_files = [get_events_file(f) for f in bold_files]

logging.info(f'Found {len(bold_files)} preprocessed bold files')

fmri_list = []
image_list = []
stim_names = []  # 保存stimulus_name用于后续检查
missing_or_dummy_count = 0
total_trials = 0
successful_match_count = 0  # 统计成功匹配真实图像的数量

# BOLD5000实际TR为2.0秒，直接硬编码
TR = 2.0

for bold_file, events_file in tqdm(zip(bold_files, events_files), total=len(bold_files)):
    if not os.path.exists(events_file):
        logging.warning(f'No events file found for {bold_file}: {events_file}')
        continue

    try:
        img = nib.load(bold_file)
        data = img.get_fdata()  # (x, y, z, t)

        events = pd.read_csv(events_file, sep='\t')

        if 'event_type' not in events.columns or 'stimulus_name' not in events.columns:
            logging.warning(f'Column missing in {events_file}')
            continue

        stim_events = events[events['event_type'] == 'stimulus']

        for _, row in stim_events.iterrows():
            total_trials += 1
            onset = row['onset']

            if not np.isfinite(onset):
                logging.warning(f'Skipping trial with non-finite onset: {onset}')
                continue

            stim_name = row['stimulus_name'].strip()

            # 搜索图像：优先大写扩展名（ImageNet 多为 .JPEG）
            img_path = None
            for sub in sub_dirs:
                for ext in ['.JPEG', '.JPG', '.jpeg', '.jpg', '']:
                    candidate = os.path.join(stim_base, sub, stim_name + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

            # 尝试加载图像
            is_real_image = False
            pil_img = Image.new('RGB', (224, 224), (255, 0, 0))  # 默认红色dummy

            if img_path and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
                    # === 修改处：放宽纯色检测阈值到 > 1（允许低对比度真实图像）===
                    img_array = np.array(pil_img)
                    if np.std(img_array) > 1 and not np.all(img_array == img_array[0, 0]):
                        is_real_image = True
                    else:
                        logging.info(f'Skipped low-contrast image (std <= 1): {stim_name} -> {img_path}')
                except UnidentifiedImageError as e:
                    logging.warning(f'Cannot identify image file {img_path}: {e}')
                except Exception as e:
                    logging.warning(f'Error opening image {img_path}: {e}')
            else:
                logging.warning(f'Missing image: {stim_name}')

            if not is_real_image:
                missing_or_dummy_count += 1
                continue  # 跳过dummy或低对比图像

            # === 成功匹配真实图像：打印名称和路径 ===
            successful_match_count += 1
            logging.info(f'[{successful_match_count}] Successfully matched real image: {stim_name} -> {img_path}')

            # fMRI提取
            start_tr = int((onset + 4) // TR)
            end_tr = int((onset + 10) // TR)

            if end_tr >= data.shape[-1] or start_tr >= end_tr:
                logging.warning(f'Skipping invalid TR range: start={start_tr}, end={end_tr}, data_len={data.shape[-1]}')
                continue

            vol = data[..., start_tr:end_tr].mean(axis=-1)
            voxels = vol.flatten()[:10000]
            voxels = np.nan_to_num(voxels, nan=0.0, posinf=0.0, neginf=0.0)

            # 保存真实配对
            fmri_list.append(voxels)
            image_list.append(pil_img)
            stim_names.append(stim_name)

    except Exception as e:
        logging.error(f'Error processing {bold_file}: {e}')
        continue

fmri_array = np.array(fmri_list)  # (N, 10000)
logging.info(f'Processed {len(fmri_array)} REAL trials (strictly no dummy)')
logging.info(f'Successfully matched {successful_match_count} real images')
logging.info(f'Skipped {missing_or_dummy_count} dummy/missing/corrupt/low-contrast out of {total_trials} total trials')

# 保存
np.save('/root/BriLLM0.5/ds001246_fmri_10k.npy', fmri_array)
np.save('/root/BriLLM0.5/ds001246_images_np.npy', np.array([np.array(img) for img in image_list]))
with open('/root/BriLLM0.5/ds001246_images_pil.pkl', 'wb') as f:
    pickle.dump(image_list, f)
with open('/root/BriLLM0.5/ds001246_stim_names.pkl', 'wb') as f:
    pickle.dump(stim_names, f)

print(f'完成！得到 {len(fmri_array)} 个真实图像配对（已严格过滤dummy）')
print(f'成功匹配真实图像数量: {successful_match_count}')
print(f'跳过 dummy/缺失/损坏/低对比图像数: {missing_or_dummy_count}（总trial: {total_trials}）')
logging.info('Preprocess finished - only real images kept (relaxed std threshold to >1)')