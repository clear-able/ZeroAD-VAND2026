"""
MVTec2 提交文件生成脚本

对 test_private 和 test_private_mixed 推理，生成 .tiff (float16) 异常图。
输出按提交格式组织目录结构。

使用示例:
    conda activate cl-ad
    cd /data/cl/MyAD

    python submit_mvtec2.py --weight ./train_log/uniadet_mvtec_final.pth --data_dir ../datasets

    python submit_mvtec2.py --weight ./train_log/uniadet_combined_final.pth --data_dir ../datasets

    python submit_mvtec2.py --weight ./train_log/uniadet_mvtec_final.pth --data_dir ../datasets --output_dir ./submission
"""

import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uniadet_model import build_uniadet


MVTEC2_CLASSES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
SPLITS = ['test_private', 'test_private_mixed']


def get_transforms(image_size):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform


def main():
    parser = argparse.ArgumentParser("MVTec2 提交文件生成")
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_dir", type=str, default="../datasets", help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./submission", help="输出目录")
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=int, default=4, help="高斯平滑 sigma")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtec2_root = os.path.join(args.data_dir, 'mvtec2')
    if not os.path.isdir(mvtec2_root):
        raise FileNotFoundError(f"找不到 MVTec2 数据集: {mvtec2_root}")

    model = build_uniadet(device=device, features_list=args.features_list)
    model.to(device)
    model.load_weights(args.weight)
    model.eval()

    img_transform = get_transforms(args.image_size)

    total = 0
    for cls_name in MVTEC2_CLASSES:
        for split in SPLITS:
            split_dir = os.path.join(mvtec2_root, cls_name, split)
            if os.path.isdir(split_dir):
                total += len([f for f in os.listdir(split_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])


    for cls_name in MVTEC2_CLASSES:
        for split in SPLITS:
            os.makedirs(os.path.join(args.output_dir, 'anomaly_images', cls_name, split), exist_ok=True)

    pbar = tqdm(total=total, desc="推理中")

    for cls_name in MVTEC2_CLASSES:
        for split in SPLITS:
            split_dir = os.path.join(mvtec2_root, cls_name, split)
            if not os.path.isdir(split_dir):
                continue

            out_dir = os.path.join(args.output_dir, 'anomaly_images', cls_name, split)

            img_files = sorted([f for f in os.listdir(split_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            for img_name in img_files:
                img_path = os.path.join(split_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = img_transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    _, anomaly_maps = model.inference(img_tensor, args.image_size)
                    anomaly_map = anomaly_maps[0].cpu().numpy()
                    anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)

                out_name = os.path.splitext(img_name)[0] + '.tiff'
                out_path = os.path.join(out_dir, out_name)
                tifffile.imwrite(out_path, anomaly_map.astype(np.float16))

                pbar.update(1)

    pbar.close()

    print("\n" + "=" * 50)
    print("=" * 50)
    for cls_name in MVTEC2_CLASSES:
        for split in SPLITS:
            out_dir = os.path.join(args.output_dir, 'anomaly_images', cls_name, split)
            n_files = len([f for f in os.listdir(out_dir) if f.endswith('.tiff')])
            print(f"  {cls_name}/{split}: {n_files} tiff files")

    print("=" * 50)


if __name__ == '__main__':
    main()
