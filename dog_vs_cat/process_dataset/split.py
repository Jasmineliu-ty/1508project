import os
import shutil
import random

def split_dataset(train_dir, test_dir, split_ratio=0.1):
    """
    将 train 文件夹中的图片按照 split_ratio 随机分配一部分到 test 文件夹。
    
    :param train_dir: 训练集文件夹路径
    :param test_dir: 测试集文件夹路径
    :param split_ratio: 分配到测试集的比例 (默认 10%)
    """
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 获取 train 文件夹下的所有文件
    files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    
    # 确保文件名包含“猫”或“狗”，根据文件名过滤
    cat_files = [f for f in files if 'cat' in f.lower()]
    dog_files = [f for f in files if 'dog' in f.lower()]

    # 按比例随机抽取
    cat_sample = random.sample(cat_files, int(len(cat_files) * split_ratio))
    dog_sample = random.sample(dog_files, int(len(dog_files) * split_ratio))

    # 将抽取的文件移动到 test 文件夹
    for file in cat_sample + dog_sample:
        shutil.move(os.path.join(train_dir, file), os.path.join(test_dir, file))

    print(f"成功将 {len(cat_sample)} 张猫图片和 {len(dog_sample)} 张狗图片移动到测试集。")

# 设置路径
train_dir = 'Train\\cats'
test_dir = 'Test\\cats'

# 调用函数
split_dataset(train_dir, test_dir, split_ratio=0.1)
