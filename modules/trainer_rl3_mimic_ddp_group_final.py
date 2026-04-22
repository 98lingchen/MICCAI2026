import copy
import logging
import os
import time
from abc import abstractmethod
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from numpy import inf
import numpy as np
from f1chexbert import F1CheXbert

from modules.optimizers import set_lr
from modules.rewards import get_self_critical_reward, init_scorer, get_absolute_score
from modules.loss import compute_loss

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# --- 分布式辅助工具 (安全版) ---
def is_main():
    """判断是否为主进程 (Rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0

def all_gather_list_safe(local_list):
    """
    安全汇总函数：处理不同 GPU 上样本数量不一致的情况。
    防止在 4 节点 (8 GPU) 评估时因为样本数不能整除而卡死。
    """
    if not dist.is_initialized():
        return local_list

    # 1. 同步所有进程的样本数量
    local_size = torch.tensor([len(local_list)], device="cuda")
    all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    # 2. 对数据进行 Padding (补齐到最大长度)
    padded_list = local_list + [None] * (max_size - len(local_list))
    
    # 3. 汇总所有对象
    gathered_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_data, padded_list)

    if is_main():
        final_list = []
        for i, sublist in enumerate(gathered_data):
            # 还原 Padding 之前的数据
            final_list.extend(sublist[:all_sizes[i]])
        return final_list
    return None

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args):
        self.args = args            

        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("\n======================================\n")
        self.logger = logging.getLogger(__name__)
    
        # 1. 设置设备与 Rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        
        # --- 优化点 1: F1CheXbert 安全加载 ---
        # 很多 BERT 类模型在 .to(device) 时会触发初始化或下载。
        # 我们增加日志并使用 barrier 确保同步。
        if is_main():
            self.logger.info("Initializing F1CheXbert on main rank...")
        
        try:
            # 务必确保已设置 export TRANSFORMERS_OFFLINE=1
            self.f1chexbert = F1CheXbert().to(self.device)
        except Exception as e:
            self.logger.error(f"Rank {self.local_rank} failed to load F1CheXbert: {e}")
            raise e

        # 屏障：确保所有进程都完成了 F1CheXbert 加载再往下走
        if dist.is_initialized():
            dist.barrier()
        if is_main(): self.logger.info("F1CheXbert loaded on all ranks.")

        # 2. 模型搬运
        self.model = model.to(self.device)

# self.model = model.to(self.device)

        # 稳健获取 tokenizer：如果已经是 DDP 就从 .module 拿，否则直接拿
        if hasattr(model, 'module'):
            self.tokenizer = model.module.tokenizer
        else:
            self.tokenizer = model.tokenizer

        # --- 优化点 2: 恢复断点顺序 ---
        # 必须在包装 DDP 之前恢复权重，否则 DDP 的参数同步可能出错
        if args.resume is not None:
            if is_main():
                self.logger.info(f"Resuming checkpoint from: {args.resume}")
            self._resume_checkpoint(args.resume)
            
            if dist.is_initialized():
                dist.barrier() # 等待主进程读完磁盘

        # 3. 模型包装与 DDP 初始化
        if dist.is_initialized():
            if is_main(): self.logger.info("Wrapping model with DDP (find_unused_parameters=True)...")
            # 注意：只有在确定模型中有部分参数不参与前向传播时才开启 find_unused_parameters
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.ve_optimizer = ve_optimizer
        self.ed_optimizer = ed_optimizer
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if is_main():
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.info(f"Trainer initialized. Start epoch: {self.start_epoch}")

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 确保 DDP 模式下 Shuffle 正常工作
            if dist.is_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            result = self._train_epoch(epoch)
            
            if is_main():
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)
                
                for key, value in log.items():
                    self.logger.info('\t{:15s}: {}'.format(str(key), value))
                
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=improved)

    def _resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        curr_has_module = any(k.startswith('module.') for k in self.model.state_dict().keys())
        ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
        
        for k, v in state_dict.items():
            if curr_has_module and not ckpt_has_module:
                new_state_dict['module.' + k] = v
            elif not curr_has_module and ckpt_has_module:
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _save_checkpoint(self, epoch, save_best=False):

        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            've_optimizer': self.ve_optimizer.state_dict(),
            'ed_optimizer': self.ed_optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        # filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        if save_best and is_main():
            torch.save(state, os.path.join(self.checkpoint_dir, 'model_best.pth'))

    def _record_best(self, log):
        # 简化版记录，实际可根据需要扩展
        pass

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args, 
                 train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        # self.tokenizer = model.tokenizer # 备份 tokenizer，防止 DDP 包装后找不到

    def _train_epoch(self, epoch):
        if is_main():
            self.logger.info(f"===== Epoch {epoch} GRPO Training Start =====")
        
        train_loss = 0
        self.model.train()
        # 对应 parser 中的 --train_sample_n
        group_size = self.args.train_sample_n 
        Bleu_scorer = Bleu(4)
        Meteor_scorer = Meteor()
        Rouge_scorer = Rouge()
        # 初始化 Scorer (主进程优先，防止死锁)
        if is_main():
            init_scorer()
        if dist.is_initialized():
            dist.barrier()
        if not is_main():
            init_scorer()
        if dist.is_initialized():
            dist.barrier()

        for batch_idx, (images_id, images, reports_ids, reports_masks, parameter_lambda) in enumerate(self.train_dataloader):
            images = images.to(self.device, non_blocking=True)
            reports_ids = reports_ids.to(self.device, non_blocking=True)
            reports_masks = reports_masks.to(self.device, non_blocking=True)
            pl_np = np.array(parameter_lambda).squeeze()
            batch_size = images.size(0)

            self.ve_optimizer.zero_grad()
            self.ed_optimizer.zero_grad()


            gen_result, sample_logprobs = self.model(
                images, mode='sample', parameter_lambda=pl_np,
                update_opts={
                    'sample_method': self.args.train_sample_method, # 'sample'
                    'sample_n': group_size,
                    'temperature': self.args.temperature 
                }
            )

            with torch.no_grad():

                gen_result_np = gen_result.detach().cpu().numpy()
                gt_ids_np = reports_ids[:, 1:].detach().cpu().numpy()
                
                gen_reports = self.tokenizer.decode_batch(gen_result_np)
                gt_reports_raw = self.tokenizer.decode_batch(gt_ids_np)
                

                gt_reports_expanded = [gt for gt in gt_reports_raw for _ in range(group_size)]
                pl_expanded = np.repeat(pl_np, group_size)


                res_final = {i: [gen_reports[i]] for i in range(len(gen_reports))}
                gts_final = {i: [gt_reports_raw[i // group_size]] for i in range(len(gen_reports))}

                _, bleu_scores = Bleu_scorer.compute_score(gts_final, res_final, verbose=0)
                bleu_4 = np.array(bleu_scores[3])

                try:
                    _, meteor_scores = Meteor_scorer.compute_score(gts_final, res_final)
                except Exception as e:
                    print(f"METEOR error: {e}")
                    meteor_scores = [0.0] * len(res_final)
                meteor_scores = np.array(meteor_scores)
                _, rouge_scores = Rouge_scorer.compute_score(gts_final, res_final)
                rouge_scores = np.array(rouge_scores)


                reward_nlp = self.args.bleu_weight * bleu_4 + self.args.meteor_weight * meteor_scores + self.args.rouge_weight * rouge_scores
                reward_nlp = torch.from_numpy(reward_nlp).to(self.device)


                _, _, p, r, _, _, _ = self.f1chexbert(hyps=gen_reports, refs=gt_reports_expanded)
                reward_p_r = torch.from_numpy(p * pl_expanded + r * (1 - pl_expanded)).to(self.device)

                total_rewards = (reward_nlp * (1 - self.args.clinical_weight) + reward_p_r * self.args.clinical_weight).view(batch_size, group_size)

                mean = total_rewards.mean(dim=1, keepdim=True)
                std = total_rewards.std(dim=1, keepdim=True)
                advantages = (total_rewards - mean) / (std + 1e-8)
                
                # 展平回 [B*G, 1] 并重复至 SeqLen 以匹配 logprobs
                advantages = advantages.view(-1, 1).repeat(1, gen_result.shape[1])

            # --- 步骤 3: RL Loss 反向传播 ---
            loss_rl = self.criterion(sample_logprobs, gen_result.data, advantages.float())
            weighted_loss_rl = 0.99 * loss_rl
            weighted_loss_rl.backward()

            # 及时释放显存
            del gen_result, sample_logprobs, advantages, total_rewards, reward_nlp, reward_p_r

            # --- 步骤 4: NLL Loss (监督学习分支) ---
            # NLL 仅对原始 Batch 进行，用于维持文本的基本结构
            output = self.model(images, parameter_lambda=pl_np, targets=reports_ids, mode='train')
            loss_nll = compute_loss(output, reports_ids, reports_masks)
            weighted_loss_nll = 0.01 * loss_nll
            weighted_loss_nll.backward()

            # --- 步骤 5: 更新模型 ---
            self.ve_optimizer.step()
            self.ed_optimizer.step()
            
            current_loss = weighted_loss_rl.item() + weighted_loss_nll.item()
            train_loss += current_loss

            if batch_idx % 10 == 0 and is_main():
                self.logger.info(f"Epoch {epoch} - Batch {batch_idx}/{len(self.train_dataloader)} - Loss: {current_loss:.4f}")

        # 记录日志并评估
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        log.update(self._evaluate_set(epoch, self.val_dataloader, "val"))
        log.update(self._evaluate_set(epoch, self.test_dataloader, "test"))

        return log


            # gen_result, sample_logprobs = self.model(
            #     images, mode='sample', parameter_lambda=pl_np,
            #     update_opts={
            #         'sample_method': self.args.train_sample_method, # 'sample'
            #         'sample_n': group_size,
            #         'temperature': self.args.temperature 
            #     }
            # )   
    def _evaluate_set(self, epoch, dataloader, split_name):
        self.model.eval()
        local_data_pairs = [] # 用于打包 (id, gt, res)
        
        with torch.no_grad():
            for batch in dataloader:
                img_ids, images, reports_ids, _, pl = batch
                pl_np = np.array(pl).squeeze()
                
                # 模型推理
                output, _ = self.model(images.to(self.device), parameter_lambda=pl_np, mode='sample', update_opts={
                    'sample_n': 1,
                })
                
                # 解码
                reports = self.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                
                # --- 改进点 1: 将 ID, GT, RES 捆绑在一起 ---
                for i in range(len(reports)):
                    # 确保 idx 是可哈希的（字符串或数字）
                    idx = img_ids[i] if isinstance(img_ids, list) else img_ids[i].item()
                    local_data_pairs.append((idx, ground_truths[i], reports[i]))

        # --- 改进点 2: 一次性汇总，避免多次汇总导致的顺序错乱 ---
        all_combined = all_gather_list_safe(local_data_pairs)

        log = {}
        # 注意：all_gather_list_safe 内部逻辑仅在 is_main 下返回完整列表
        if is_main():
            # --- 改进点 3: 使用字典去重并强制对齐 ---
            # 这能自动解决 DDP DistributedSampler 补齐产生的重复样本问题
            unique_data = {}
            for img_id, gt, res in all_combined:
                unique_data[img_id] = {"gt": gt, "res": res}
            
            # 为了保证每次评估的顺序一致（方便对比），进行排序
            sorted_ids = sorted(unique_data.keys())
            
            # 构造符合 pycocoevalcap 要求的格式：{index: [text]}
            # 此时 final_gts 和 final_res 的 keys 来源完全一致，绝对不会报 AssertionError
            final_gts = {i: [unique_data[k]["gt"]] for i, k in enumerate(sorted_ids)}
            final_res = {i: [unique_data[k]["res"]] for i, k in enumerate(sorted_ids)}

            # 计算指标
            metrics = self.metric_ftns(final_gts, final_res, use_clinical=True)
            log = {f"{split_name}_{k}": v for k, v in metrics.items()}
            
            # 保存测试结果
            if split_name == "test":
                df = pd.DataFrame([
                    {"image_id": k, "ground_truth": unique_data[k]["gt"], "prediction": unique_data[k]["res"]} 
                    for k in sorted_ids
                ])
                df.to_excel(os.path.join(self.checkpoint_dir, f"{epoch}_test.xlsx"), index=False)
        
        # 必须同步，防止 Rank 0 计算太慢导致其他 Rank 超时挂掉
        if dist.is_initialized():
            dist.barrier() 
            
        # 广播 log 给所有进程，防止非主进程返回 None 导致后续 update(None) 崩溃
        # 如果你的外层逻辑已经处理了非主进程不读 log，则不需要广播
        return log