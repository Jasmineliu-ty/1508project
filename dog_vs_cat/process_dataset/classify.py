import os
import shutil

def copy_files_with_cat(source_folder, target_folder):
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(source_folder):
        if 'cat' in file_name:  # 判断文件名中是否包含 'cat'
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)
            
            # 检查是否为文件（排除文件夹）
            if os.path.isfile(source_path):
                shutil.copy(source_path, target_path)  # 复制文件
                print(f"Copied: {source_path} -> {target_path}")
    print("File copying complete.")

# 示例使用
source_folder = r'C:\\Users\\HP\\Desktop\\dogs-vs-cats\\test1\\test1'  # 替换为源文件夹路径
target_folder = r'Test\\cats'  # 替换为目标文件夹路径
copy_files_with_cat(source_folder, target_folder)