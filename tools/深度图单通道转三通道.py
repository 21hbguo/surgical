"""模块说明：负责当前主要逻辑。"""

# 工具模块：提供 preprocess_depth_3c 相关的通用辅助能力。

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from data import _normalize_array, depth_to_rgb


# 处理深度相关逻辑。
def process_depth_file(input_path, output_path):
    """将单通道深度图转换为3通道并保存"""
    # 读取单通道深度图
    depth = np.array(Image.open(input_path))
    
    # 归一化
    depth_normalized = _normalize_array(depth, method="minmax")
    
    # 转换为3通道深度图
    depth3 = depth_to_rgb(depth_normalized)
    
    # 转换为0-255范围
    depth3 = (depth3 * 255).astype(np.uint8)
    
    # 保存为PNG
    Image.fromarray(depth3).save(output_path)
# 组织脚本主流程。


def main():
    # 输入输出路径
    input_dir = "/home/guo/project/ssl4mis/data/endovis2017/data/depth_slices"
    output_dir = "/home/guo/project/ssl4mis/data/endovis2017/data/depth3c_slices"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有深度图文件
    depth_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    print(f"Found {len(depth_files)} depth files to process")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # 处理每个文件
    for filename in tqdm(depth_files, desc="Processing depth files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(output_path):
            continue
        
        try:
            process_depth_file(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nDone! Processed {len(depth_files)} files.")
    print(f"3-channel depth maps saved to: {output_dir}")


if __name__ == "__main__":
    main()
