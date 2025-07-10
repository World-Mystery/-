import os
import shutil
import random


def sample_and_move(src_dir, dst_dir, sample_ratio=0.15):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取源目录下所有文件（不包括子目录）
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # 计算抽样数量
    sample_size = max(1, int(len(files) * sample_ratio))  # 至少抽1个
    sampled_files = random.sample(files, sample_size)

    # 移动文件
    for file_name in sampled_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} -> {dst_path}")


# 设置路径
train_hc = 'autodl-tmp/train/HC'
train_patient = 'autodl-tmp/train/patient'
test_hc = 'autodl-tmp/test/HC'
test_patient = 'autodl-tmp/test/patient'

if __name__ == "__main__":
    sample_and_move(train_hc, test_hc)
    sample_and_move(train_patient, test_patient)