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


def is_main():

    return not dist.is_initialized() or dist.get_rank() == 0

def all_gather_list_safe(local_list):

    if not dist.is_initialized():
        return local_list


    local_size = torch.tensor([len(local_list)], device="cuda")
    all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)


    padded_list = local_list + [None] * (max_size - len(local_list))
    

    gathered_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_data, padded_list)

    if is_main():
        final_list = []
        for i, sublist in enumerate(gathered_data):

            final_list.extend(sublist[:all_sizes[i]])
        return final_list
    return None

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args):
        self.args = args            
        self.logger = logging.getLogger(__name__)
    

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        

        if is_main():
            self.logger.info("Initializing F1CheXbert on main rank...")
        
        try:

            self.f1chexbert = F1CheXbert().to(self.device)
        except Exception as e:
            self.logger.error(f"Rank {self.local_rank} failed to load F1CheXbert: {e}")
            raise e


        if dist.is_initialized():
            dist.barrier()
        if is_main(): self.logger.info("F1CheXbert loaded on all ranks.")


        self.model = model.to(self.device)


        if hasattr(model, 'module'):
            self.tokenizer = model.module.tokenizer
        else:
            self.tokenizer = model.tokenizer


        if args.resume is not None:
            if is_main():
                self.logger.info(f"Resuming checkpoint from: {args.resume}")
            self._resume_checkpoint(args.resume)
            
            if dist.is_initialized():
                dist.barrier() 


        if dist.is_initialized():
            if is_main(): self.logger.info("Wrapping model with DDP (find_unused_parameters=True)...")

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

        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        if save_best and is_main():
            torch.save(state, os.path.join(self.checkpoint_dir, 'model_best.pth'))

    def _record_best(self, log):

        pass

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args, 
                 train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


    def _train_epoch(self, epoch):
        if is_main():
            self.logger.info(f"===== Epoch {epoch} GRPO Training Start =====")
        
        train_loss = 0
        self.model.train()

        group_size = self.args.train_sample_n 
        Bleu_scorer = Bleu(4)
        Meteor_scorer = Meteor()
        Rouge_scorer = Rouge()

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


                reward_nlp = 5/11 * bleu_4 + 1/11 * meteor_scores + 5/11 * rouge_scores
                reward_nlp = torch.from_numpy(reward_nlp).to(self.device)



                _, _, p, r, _, _, _ = self.f1chexbert(hyps=gen_reports, refs=gt_reports_expanded)
                reward_p_r = torch.from_numpy(p * pl_expanded + r * (1 - pl_expanded)).to(self.device)


                total_rewards = (reward_nlp * 0.9 + reward_p_r * 0.1).view(batch_size, group_size)

                mean = total_rewards.mean(dim=1, keepdim=True)
                std = total_rewards.std(dim=1, keepdim=True)
                advantages = (total_rewards - mean) / (std + 1e-8)
                

                advantages = advantages.view(-1, 1).repeat(1, gen_result.shape[1])

            loss_rl = self.criterion(sample_logprobs, gen_result.data, advantages.float())
            weighted_loss_rl = 0.99 * loss_rl
            weighted_loss_rl.backward()

            del gen_result, sample_logprobs, advantages, total_rewards, reward_nlp, reward_p_r

            output = self.model(images, parameter_lambda=pl_np, targets=reports_ids, mode='train')
            loss_nll = compute_loss(output, reports_ids, reports_masks)
            weighted_loss_nll = 0.01 * loss_nll
            weighted_loss_nll.backward()


            self.ve_optimizer.step()
            self.ed_optimizer.step()
            
            current_loss = weighted_loss_rl.item() + weighted_loss_nll.item()
            train_loss += current_loss

            if batch_idx % 10 == 0 and is_main():
                self.logger.info(f"Epoch {epoch} - Batch {batch_idx}/{len(self.train_dataloader)} - Loss: {current_loss:.4f}")

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        log.update(self._evaluate_set(epoch, self.val_dataloader, "val"))
        log.update(self._evaluate_set(epoch, self.test_dataloader, "test"))

        return log


    def _evaluate_set(self, epoch, dataloader, split_name):
        self.model.eval()
        local_data_pairs = [] 
        
        with torch.no_grad():
            for batch in dataloader:
                img_ids, images, reports_ids, _, pl = batch
                pl_np = np.array(pl).squeeze()
                
            
                output, _ = self.model(images.to(self.device), parameter_lambda=pl_np, mode='sample', update_opts={
                    'sample_n': 1,
                })
                
                
                reports = self.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                

                for i in range(len(reports)):

                    idx = img_ids[i] if isinstance(img_ids, list) else img_ids[i].item()
                    local_data_pairs.append((idx, ground_truths[i], reports[i]))


        all_combined = all_gather_list_safe(local_data_pairs)

        log = {}

        if is_main():

            unique_data = {}
            for img_id, gt, res in all_combined:
                unique_data[img_id] = {"gt": gt, "res": res}
            

            sorted_ids = sorted(unique_data.keys())
            

            final_gts = {i: [unique_data[k]["gt"]] for i, k in enumerate(sorted_ids)}
            final_res = {i: [unique_data[k]["res"]] for i, k in enumerate(sorted_ids)}


            metrics = self.metric_ftns(final_gts, final_res, use_clinical=True)
            log = {f"{split_name}_{k}": v for k, v in metrics.items()}
            

            if split_name == "test":
                df = pd.DataFrame([
                    {"image_id": k, "ground_truth": unique_data[k]["gt"], "prediction": unique_data[k]["res"]} 
                    for k in sorted_ids
                ])
                df.to_excel(os.path.join(self.checkpoint_dir, f"{epoch}_test.xlsx"), index=False)
        

        if dist.is_initialized():
            dist.barrier() 
            
        return log