"""
UniADet 训练和推理入口脚本

使用示例:
    python train_uniadet.py --dataset visa --test_dataset mvtec --data_dir ../datasets

    python train_uniadet.py --dataset visa --test_dataset mvtec --weight ./checkpoints/xxx.pth
"""

import os
import sys
import argparse
import logging
import random
import json
import multiprocessing as _mp
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure

from uniadet_model import build_uniadet


# ============================================================
# ============================================================

DATASET_CLASSES = {
    'mvtec': ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
              'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
             'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
}

DATASET_DIR_NAMES = {
    'mvtec': ['mvtec', 'mvtec_ad', 'mvtec_anomaly_detection'],
    'visa': ['visa', 'VisA'],
}


def find_dataset_root(data_dir, dataset_name):
    """查找数据集实际目录"""
    possible_names = DATASET_DIR_NAMES.get(dataset_name, [dataset_name])
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"找不到数据集目录，尝试了: {possible_names} 在 {data_dir}")


def generate_class_info(dataset_name):
    """生成类别信息"""
    if dataset_name not in DATASET_CLASSES:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    obj_list = DATASET_CLASSES[dataset_name]
    class_name_map_class_id = {k: i for i, k in enumerate(obj_list)}
    return obj_list, class_name_map_class_id


class AnomalyDataset(Dataset):
    """
    异常检测数据集 - 支持两种加载方式：
    1. 如果存在 meta.json，使用 AnomalyCLIP 格式加载
    2. 否则从标准目录结构加载

    标准目录结构 (MVTec):
        dataset_root/
        ├── category1/
        │   ├── train/good/
        │   ├── test/good/
        │   ├── test/defect_type/
        │   └── ground_truth/defect_type/
        └── ...
    """

    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

        meta_path = os.path.join(root, 'meta.json')
        if os.path.exists(meta_path):
            self._load_from_meta_json(meta_path, mode)
        else:
            self._load_from_directory(mode)


    def _load_from_meta_json(self, meta_path, mode):
        """从 meta.json 加载（AnomalyCLIP 格式）"""
        meta_info = json.load(open(meta_path, 'r'))
        meta_info = meta_info[mode]

        for cls_name in meta_info.keys():
            for item in meta_info[cls_name]:
                self.data_all.append({
                    'img_path': os.path.join(self.root, item['img_path']),
                    'mask_path': os.path.join(self.root, item['mask_path']) if item.get('mask_path') else None,
                    'cls_name': item['cls_name'],
                    'anomaly': item['anomaly'],
                    'defect_type': item.get('specie_name', 'unknown')
                })

    def _load_from_directory(self, mode):
        """从标准目录结构加载"""

        for cls_name in self.obj_list:
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            if mode == 'test':
                test_dir = os.path.join(cls_dir, 'test')
                gt_dir = os.path.join(cls_dir, 'ground_truth')

                if not os.path.isdir(test_dir):
                    continue

                for defect_type in sorted(os.listdir(test_dir)):
                    defect_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    is_good = defect_type.lower() == 'good'

                    for img_name in sorted(os.listdir(defect_dir)):
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            continue

                        img_path = os.path.join(defect_dir, img_name)

                        mask_path = None
                        if not is_good:
                            mask_name_base = os.path.splitext(img_name)[0]
                            for mask_ext in ['_mask.png', '.png', '_mask.jpg', '.jpg']:
                                candidate = os.path.join(gt_dir, defect_type, mask_name_base + mask_ext)
                                if os.path.isfile(candidate):
                                    mask_path = candidate
                                    break

                        self.data_all.append({
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'cls_name': cls_name,
                            'anomaly': 0 if is_good else 1,
                            'defect_type': defect_type
                        })

            else:  # train mode
                train_dir = os.path.join(cls_dir, 'train', 'good')
                if not os.path.isdir(train_dir):
                    continue

                for img_name in sorted(os.listdir(train_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue

                    img_path = os.path.join(train_dir, img_name)
                    self.data_all.append({
                        'img_path': img_path,
                        'mask_path': None,
                        'cls_name': cls_name,
                        'anomaly': 0,
                        'defect_type': 'good'
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

        # Transform
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'cls_id': self.class_name_map_class_id.get(cls_name, 0)
        }


def get_transforms(image_size):
    """获取数据预处理 transform"""
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    return img_transform, mask_transform


# ============================================================
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logit, target):
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1).long()

        # One-hot encoding
        one_hot = torch.zeros_like(logit)
        one_hot.scatter_(1, target, 1)

        # Softmax
        pt = (one_hot * logit).sum(1) + 1e-5
        logpt = pt.log()

        # Focal weight
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        return loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        N = target.size(0)
        pred_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = (pred_flat * target_flat).sum(1)
        dice = (2 * intersection + self.smooth) / (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)

        return 1 - dice.mean()


# ============================================================
# ============================================================

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    """计算 PRO 分数"""
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step

    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean() if pro else 0)
        fprs.append(fpr)

    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < expect_fpr
    if idxes.sum() < 2:
        return 0

    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min() + 1e-8)
    from sklearn.metrics import auc
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def compute_metrics_for_obj(obj_data):
    """计算单个类别的指标（全局函数，支持 multiprocessing）"""
    obj, data = obj_data
    gt_sp = np.array(data['gt_sp'])
    pr_sp = np.array(data['pr_sp'])
    gt_px = np.stack(data['imgs_masks'])
    pr_px = np.stack(data['anomaly_maps'])

    i_auroc = roc_auc_score(gt_sp, pr_sp) if len(np.unique(gt_sp)) > 1 else 0
    i_ap = average_precision_score(gt_sp, pr_sp) if len(np.unique(gt_sp)) > 1 else 0

    if gt_px.ndim == 4:
        gt_px = gt_px.squeeze(1)
    p_auroc = roc_auc_score(gt_px.ravel(), pr_px.ravel()) if gt_px.sum() > 0 else 0
    p_aupro = cal_pro_score(gt_px, pr_px) if gt_px.sum() > 0 else 0

    return obj, i_auroc, i_ap, p_auroc, p_aupro


# ============================================================
# ============================================================

def get_logger(save_path, dataset=None, test_dataset=None):
    """获取日志记录器"""
    os.makedirs(save_path, exist_ok=True)
    if dataset and test_dataset:
        txt_path = os.path.join(save_path, f'log_{dataset}_to_{test_dataset}.txt')
    else:
        txt_path = os.path.join(save_path, 'log.txt')

    log_name = f'uniadet_{dataset}_to_{test_dataset}' if (dataset and test_dataset) else 'uniadet'
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_summary_logger(save_path, dataset=None, test_dataset=None):
    """获取精简日志记录器（仅训练损失和最终指标）"""
    os.makedirs(save_path, exist_ok=True)
    if dataset and test_dataset:
        txt_path = os.path.join(save_path, f'summary_{dataset}_to_{test_dataset}.txt')
    else:
        txt_path = os.path.join(save_path, 'summary.txt')

    log_name = f'uniadet_summary_{dataset}_to_{test_dataset}' if (dataset and test_dataset) else 'uniadet_summary'
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ============================================================

def train(args):
    """训练函数"""
    logger = get_logger(args.log_dir, args.dataset, args.test_dataset)
    summary_logger = get_summary_logger(args.log_dir, args.dataset, args.test_dataset)
    logger.info("=" * 60)
    summary_logger.info("=" * 60)
    summary_logger.info("UniADet 训练摘要")
    summary_logger.info(f"训练数据集: {args.dataset} | 测试数据集: {args.test_dataset}")
    summary_logger.info("=" * 60)
    logger.info("UniADet 训练")
    logger.info(f"训练数据集: {args.dataset}")
    logger.info(f"测试数据集: {args.test_dataset}")
    logger.info(f"特征提取层: {args.features_list}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epoch}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_transform, mask_transform = get_transforms(args.image_size)

    train_data_path = find_dataset_root(args.data_dir, args.dataset)
    train_data = AnomalyDataset(
        root=train_data_path,
        transform=img_transform,
        target_transform=mask_transform,
        dataset_name=args.dataset,
        mode='test'
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = build_uniadet(
        device=device,
        features_list=args.features_list,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = DiceLoss()
    lam = 4

    for epoch in range(1, args.epoch + 1):
        model.train()
        loss_list = []
        image_loss_list = []
        seg_loss_list = []

        for items in train_dataloader:
            image = items['img'].to(device)
            label = items['anomaly'].to(device)
            gt = items['img_mask'].squeeze(1).to(device)

            gt = (gt > 0.5).float()

            cls_logits_list, seg_logits_list, cls_probs_list, seg_probs_list = model(image, args.image_size)

            image_loss = 0
            for cls_logits in cls_logits_list:
                image_loss += F.cross_entropy(cls_logits, label.long())

            seg_loss = 0
            for seg_logits in seg_logits_list:
                seg_probs = F.softmax(seg_logits, dim=1)
                seg_loss += loss_focal(seg_probs, gt)
                seg_loss += loss_dice(seg_probs[:, 1, :, :], gt)
                seg_loss += loss_dice(seg_probs[:, 0, :, :], 1 - gt)

            total_loss = image_loss + lam * seg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.item())
            image_loss_list.append(image_loss.item())
            seg_loss_list.append(seg_loss.item())

        logger.info(
            f'Epoch [{epoch}/{args.epoch}] '
            f'Loss: {np.mean(loss_list):.4f}, '
            f'Image Loss: {np.mean(image_loss_list):.4f}, '
            f'Seg Loss: {np.mean(seg_loss_list):.4f}'
        )
        summary_logger.info(
            f'Epoch [{epoch}/{args.epoch}] '
            f'Loss: {np.mean(loss_list):.4f}, '
            f'Image Loss: {np.mean(image_loss_list):.4f}, '
            f'Seg Loss: {np.mean(seg_loss_list):.4f}'
        )

        if epoch % args.save_freq == 0 or epoch == args.epoch:
            save_path = os.path.join(args.log_dir, f'uniadet_{args.dataset}_epoch{epoch}.pth')
            model.save_weights(save_path)
            logger.info(f'模型已保存: {save_path}')

    final_path = os.path.join(args.log_dir, f'uniadet_{args.dataset}_final.pth')
    model.save_weights(final_path)
    logger.info(f'最终模型已保存: {final_path}')

    logger.info("=" * 60)
    logger.info("开始在测试数据集上评估")
    test(args, model, logger, summary_logger)


def test(args, model=None, logger=None, summary_logger=None):
    """测试/推理函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if logger is None:
        logger = get_logger(args.log_dir, args.dataset, args.test_dataset)
    if summary_logger is None:
        summary_logger = get_summary_logger(args.log_dir, args.dataset, args.test_dataset)

    img_transform, mask_transform = get_transforms(args.image_size)

    test_data_path = find_dataset_root(args.data_dir, args.test_dataset)
    test_data = AnomalyDataset(
        root=test_data_path,
        transform=img_transform,
        target_transform=mask_transform,
        dataset_name=args.test_dataset,
        mode='test'
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    obj_list = test_data.obj_list

    if model is None:
        model = build_uniadet(
            device=device,
            features_list=args.features_list,
        )
        model.to(device)
        if args.weight is not None:
            model.load_weights(args.weight)
            logger.info(f'加载权重: {args.weight}')
        else:
            logger.warning('未指定权重文件')

    model.eval()

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    for items in test_dataloader:
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask'].numpy()
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        results[cls_name]['imgs_masks'].append(gt_mask[0])
        results[cls_name]['gt_sp'].append(items['anomaly'].item())

        with torch.no_grad():
            image_scores, anomaly_maps = model.inference(image, args.image_size)

            results[cls_name]['pr_sp'].append(image_scores[0].cpu().item())

            anomaly_map = anomaly_maps[0].cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)
            results[cls_name]['anomaly_maps'].append(anomaly_map)

    logger.info("计算评估指标（多核并行）...")
    obj_data_list = [(obj, results[obj]) for obj in obj_list]
    n_workers = min(6, len(obj_list))

    with _mp.get_context('spawn').Pool(n_workers) as pool:
        metrics_results = pool.map(compute_metrics_for_obj, obj_data_list)

    table = []
    i_auroc_list, i_ap_list, p_auroc_list, p_aupro_list = [], [], [], []

    for obj, i_auroc, i_ap, p_auroc, p_aupro in metrics_results:
        table.append([obj, f'{i_auroc*100:.1f}', f'{i_ap*100:.1f}', f'{p_auroc*100:.1f}', f'{p_aupro*100:.1f}'])
        i_auroc_list.append(i_auroc)
        i_ap_list.append(i_ap)
        p_auroc_list.append(p_auroc)
        p_aupro_list.append(p_aupro)

    table.append(['Mean', f'{np.mean(i_auroc_list)*100:.1f}', f'{np.mean(i_ap_list)*100:.1f}',
                  f'{np.mean(p_auroc_list)*100:.1f}', f'{np.mean(p_aupro_list)*100:.1f}'])

    from tabulate import tabulate
    result_str = tabulate(table, headers=['Object', 'I-AUROC', 'I-AP', 'P-AUROC', 'P-AUPRO'], tablefmt='pipe')
    logger.info(f"\n{result_str}")

    logger.info("=" * 60)
    logger.info(f"Image AUROC: {np.mean(i_auroc_list)*100:.1f}%")
    logger.info(f"Image AP:    {np.mean(i_ap_list)*100:.1f}%")
    logger.info(f"Pixel AUROC: {np.mean(p_auroc_list)*100:.1f}%")
    logger.info(f"Pixel AUPRO: {np.mean(p_aupro_list)*100:.1f}%")
    logger.info("=" * 60)

    summary_logger.info("评估结果（Mean）")
    summary_logger.info(f"Image AUROC: {np.mean(i_auroc_list)*100:.1f}%")
    summary_logger.info(f"Image AP:    {np.mean(i_ap_list)*100:.1f}%")
    summary_logger.info(f"Pixel AUROC: {np.mean(p_auroc_list)*100:.1f}%")
    summary_logger.info(f"Pixel AUPRO: {np.mean(p_aupro_list)*100:.1f}%")
    summary_logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser("UniADet")

    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--log_dir", type=str, default="./train_log")

    parser.add_argument("--dataset", type=str, default="visa", choices=['mvtec', 'visa'])
    parser.add_argument("--test_dataset", type=str, default="mvtec", choices=['mvtec', 'visa'])

    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    parser.add_argument("--image_size", type=int, default=518)

    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=111)

    parser.add_argument("--sigma", type=int, default=4)

    parser.add_argument("--weight", type=str, default=None)

    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    if args.weight is None:
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
