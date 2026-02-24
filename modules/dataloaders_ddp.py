import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset



class R2DataLoader(DataLoader):
    """
    DDP-ready DataLoader:
      - 在多节点/多GPU环境下自动分配数据子集
      - 确保每个 epoch 的数据打乱在不同节点间是同步的
    """
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # ---------------- 1. 数据增强 (Transforms) ----------------
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        # ---------------- 2. 数据集实例化 ----------------
        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        # ---------------- 3. 分布式采样器 (Distributed Sampler) ----------------
        self.distributed = dist.is_available() and dist.is_initialized()
        
        if self.distributed:
            # 获取当前节点的 rank 信息
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.shuffle,  # 只有 train split 会是 True
                drop_last=(split == 'train') # 训练集设为 True 防止 batch 不齐导致 DDP 挂起
            )
            # 使用 sampler 时，DataLoader 必须设置 shuffle=False
            shuffle_flag = False
        else:
            self.sampler = None
            shuffle_flag = self.shuffle

        # ---------------- 4. 性能优化配置 ----------------
        # 在多节点环境下，提高 num_workers 和 pin_memory 至关重要
        pin_memory = True if torch.cuda.is_available() else False
        
        # persistent_workers 在 epoch 切换时不需要重新销毁并启动 worker 进程，能节省大量时间
        persistent_workers = True if self.num_workers > 0 else False
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': shuffle_flag,
            'sampler': self.sampler,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'drop_last': (split == 'train' and not self.distributed) # 非分布式环境的 drop_last
        }

        # prefetch_factor: 每个 worker 提前加载的 batch 数
        if self.num_workers > 0:
            self.init_kwargs['prefetch_factor'] = 2

        super().__init__(**self.init_kwargs)

    def set_epoch(self, epoch: int):
        """关键：在 Trainer 的每个 epoch 开始前调用，保证 shuffle 随机性"""
        if self.distributed and self.sampler is not None:
            self.sampler.set_epoch(epoch)

    @staticmethod
    def collate_fn(data):
        # 保持你原有的 collate 逻辑，确保 padding 长度正确
        images_id, images, reports_ids, reports_masks, seq_lengths, parameter_lambda = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        # 返回 Tensor，方便后续直接 .to(device)
        return (images_id, 
                images, 
                torch.LongTensor(targets), 
                torch.FloatTensor(targets_masks), 
                parameter_lambda)