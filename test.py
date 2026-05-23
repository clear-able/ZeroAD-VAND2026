"""
在 MVTec2 test_public 上测试零样本模型的 SegF1（像素级 F1max）

使用示例:
    conda activate cl-ad
    cd /data/cl/MyAD

    python test_mvtec2.py --weight ./train_log/uniadet_visa_final.pth --data_dir ../datasets

    python test_mvtec2.py --weight ./train_log/uniadet_mvtec_final.pth --data_dir ../datasets

    python test_mvtec2.py --weight ./train_log/uniadet_combined_final.pth --data_dir ../datasets
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uniadet_model import build_uniadet


# ============================================================
# ============================================================

MVTEC2_CLASSES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']


class MVTec2TestPublicDataset(Dataset):
    """
    MVTec2 test_public 数据集

    目录结构:
        mvtec2/{category}/test_public/good/          正常图
        mvtec2/{category}/test_public/bad/           异常图
        mvtec2/{category}/test_public/ground_truth/bad/  异常mask
    """

    def __init__(self, root, transform, mask_transform):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_all = []

        for cls_name in MVTEC2_CLASSES:
            test_dir = os.path.join(root, cls_name, 'test_public')
            if not os.path.isdir(test_dir):
                continue

            good_dir = os.path.join(test_dir, 'good')
            if os.path.isdir(good_dir):
                for img_name in sorted(os.listdir(good_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    self.data_all.append({
                        'img_path': os.path.join(good_dir, img_name),
                        'mask_path': None,
                        'cls_name': cls_name,
                        'anomaly': 0,
                    })

            bad_dir = os.path.join(test_dir, 'bad')
            gt_dir = os.path.join(test_dir, 'ground_truth', 'bad')
            if os.path.isdir(bad_dir):
                for img_name in sorted(os.listdir(bad_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue

                    img_path = os.path.join(bad_dir, img_name)

                    mask_path = None
                    mask_name_base = os.path.splitext(img_name)[0]
                    for mask_ext in ['_mask.png', '_mask.jpg', '.png', '.jpg']:
                        candidate = os.path.join(gt_dir, mask_name_base + mask_ext)
                        if os.path.isfile(candidate):
                            mask_path = candidate
                            break

                    self.data_all.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'cls_name': cls_name,
                        'anomaly': 1,
                    })


    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = data['img_path']
        mask_path = data['mask_path']
        cls_name = data['cls_name']
        anomaly = data['anomaly']

        img = Image.open(img_path).convert('RGB')

        if anomaly == 0 or mask_path is None:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
        else:
            if os.path.isfile(mask_path):
                img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            img_mask = self.mask_transform(img_mask)

        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
        }


# ============================================================
# ============================================================

def compute_segf1(gt_masks, pred_maps):
    """计算像素级最大 F1 分数 (SegF1)"""
    gt = np.array(gt_masks).ravel()
    pred = np.array(pred_maps).ravel()
    precision, recall, _ = precision_recall_curve(gt, pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.max(f1_scores))


# ============================================================
# ============================================================

def test_mvtec2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])

    mvtec2_root = os.path.join(args.data_dir, 'mvtec2')
    if not os.path.isdir(mvtec2_root):
        raise FileNotFoundError(f"找不到 MVTec2 数据集: {mvtec2_root}")

    dataset = MVTec2TestPublicDataset(mvtec2_root, img_transform, mask_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = build_uniadet(device=device, features_list=args.features_list)
    model.to(device)
    model.load_weights(args.weight)
    model.eval()

    results = {cls: {'imgs_masks': [], 'anomaly_maps': []} for cls in MVTEC2_CLASSES}

    for items in tqdm(dataloader, desc="推理中"):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask'].numpy()
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        results[cls_name]['imgs_masks'].append(gt_mask[0])

        with torch.no_grad():
            _, anomaly_maps = model.inference(image, args.image_size)
            anomaly_map = anomaly_maps[0].cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)
            results[cls_name]['anomaly_maps'].append(anomaly_map)

    print("\n" + "=" * 50)
    print("=" * 50)
    print(f"{'Category':<16} {'SegF1':>8}")
    print("-" * 26)

    segf1_list = []
    for cls_name in MVTEC2_CLASSES:
        data = results[cls_name]
        if len(data['imgs_masks']) == 0:
            print(f"{cls_name:<16} {'N/A':>8}")
            continue

        gt_px = np.stack(data['imgs_masks'])
        pr_px = np.stack(data['anomaly_maps'])

        if gt_px.ndim == 4:
            gt_px = gt_px.squeeze(1)

        if gt_px.sum() > 0:
            segf1 = compute_segf1(gt_px, pr_px)
        else:
            segf1 = 0.0

        segf1_list.append(segf1)
        print(f"{cls_name:<16} {segf1*100:>7.1f}%")

    mean_segf1 = np.mean(segf1_list) if segf1_list else 0.0
    print("-" * 26)
    print(f"{'Mean':<16} {mean_segf1*100:>7.1f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser("MVTec2 SegF1 测试")
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_dir", type=str, default="../datasets", help="数据集根目录")
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=int, default=4, help="高斯平滑 sigma")
    args = parser.parse_args()
    test_mvtec2(args)


if __name__ == '__main__':
    main()
