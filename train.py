import os
import sys
import torch
import argparse
import logging
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.tokenizers import Tokenizer
from modules.dataloaders_ddp import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_plateau_optimizer
from modules.trainer import Trainer
from modules.loss import RewardCriterion
from models.r2gen3 import R2GenModel


def setup_logging(args):
    log_format = f"[Rank {os.environ.get('RANK', '0')}] %(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--ann_path', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)

    # Model config
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)

    # Sampling
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='results/MIMIC-CXR')
    parser.add_argument('--record_dir', type=str, default='results/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=1)
    parser.add_argument('--log_period', type=int, default=100)
    parser.add_argument('--sc_eval_period', type=int, default=10000)

    # RM
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=1e-6)
    parser.add_argument('--lr_ed', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--adam_eps', type=float, default=1e-9)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--noamopt_warmup', type=int, default=1000)
    parser.add_argument('--noamopt_factor', type=int, default=1)
    parser.add_argument('--reduce_on_plateau_factor', type=float, default=0.5)
    parser.add_argument('--reduce_on_plateau_patience', type=int, default=3)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=np.random.randint(10000))

    # Self-Critical
    parser.add_argument('--train_sample_n', type=int, default=5)
    parser.add_argument('--train_sample_method', type=str, default='sample')
    parser.add_argument('--train_beam_size', type=int, default=1)
    parser.add_argument('--sc_sample_method', type=str, default='greedy')
    parser.add_argument('--sc_beam_size', type=int, default=1)

# Distributed
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')




    return parser.parse_args()

def init_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
        
        dist.init_process_group(backend=args.dist_backend)
        
        if dist.is_initialized():
             print(f"Rank {args.rank} successfully initialized process group.", flush=True)
        
        torch.cuda.set_device(args.local_rank)
        
        dist.barrier() 
    else:
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1

def main():
    args = parse_args()
    logger = setup_logging(args)


    init_distributed(args)
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")


    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



    tokenizer = Tokenizer(args)



    train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_loader   = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_loader  = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    if args.distributed: dist.barrier()



    model = R2GenModel(args, tokenizer).to(device)


    if args.distributed:

        model = DDP(model, 
                    device_ids=[args.local_rank], 
                    output_device=args.local_rank,
                    find_unused_parameters=True)
        dist.barrier()



    criterion = RewardCriterion()
    metrics = compute_scores

    if args.distributed:
        ve_optimizer, ed_optimizer = build_plateau_optimizer(args, model.module)
    else:
        ve_optimizer, ed_optimizer = build_plateau_optimizer(args, model)



    trainer = Trainer(model.module if hasattr(model, 'module') else model, 
                    criterion, metrics, ve_optimizer, ed_optimizer, args,
                    train_loader, val_loader, test_loader)


    trainer.train()

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
