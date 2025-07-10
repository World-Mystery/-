import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

# 设置目标形状和分辨率
target_shape = (64, 64, 64)
target_resolution = (2.0, 2.0, 2.0)
target_affine = np.diag([*target_resolution, 1.0])
target_img = nib.Nifti1Image(np.zeros(target_shape), target_affine)

def preprocess_and_save(src_path, dst_path):
    try:
        img = nib.load(src_path)
        resampled_img = resample_to_img(
            img, target_img, interpolation='continuous',
            force_resample=True, copy_header=True
        )
        data = resampled_img.get_fdata().astype(np.float32)
        np.save(dst_path, data)
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")

def process_folder(src_dir, dst_dir, label):
    os.makedirs(dst_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith('.nii'):
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file.replace('.nii', '.npy'))
            preprocess_and_save(src_path, dst_path)

def main():
    base_raw = './autodl-tmp'
    base_processed = './autodl-tmp'

    # 处理训练集
    process_folder(os.path.join(base_raw, 'HC'), os.path.join(base_processed, 'train/HC'), 0)
    process_folder(os.path.join(base_raw, 'patient'), os.path.join(base_processed, 'train/patient'), 1)

    # 处理测试集
    process_folder(os.path.join(base_raw, 'test_data/HC'), os.path.join(base_processed, 'test/HC'), 0)
    process_folder(os.path.join(base_raw, 'test_data/patient'), os.path.join(base_processed, 'test/patient'), 1)

    print("预处理完成！")

if __name__ == '__main__':
    main()