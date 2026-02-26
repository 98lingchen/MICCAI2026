import os
import torch
import argparse
import numpy as np
import pandas as pd

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from models.r2gen import R2GenModel


def parse_args():
    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--ann_path', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['mimic_cxr', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=100)

    # Model config (must match training)
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

    # Sampling (used in model(mode='sample'))
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # RM (must match training)
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)


    parser.add_argument('--resume', type=str, default='',
                        help='checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='',
                        help='where to save xlsx and logs')
    parser.add_argument('--xlsx_name', type=str, default='')

    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()


def prepare_device(n_gpu_use: int):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    return device


def load_checkpoint(model, resume_path, device):
    ckpt = torch.load(resume_path, map_location=device)


    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt


    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if len(ckpt_keys) > 0:
        ckpt_has_module = ckpt_keys[0].startswith('module.')
        model_has_module = model_keys[0].startswith('module.')
        if ckpt_has_module and not model_has_module:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        elif (not ckpt_has_module) and model_has_module:
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)



@torch.no_grad()
def test_only(model, tokenizer, test_loader, device, save_dir, xlsx_name, current_lambda):
    model.eval()

    test_gts, test_res, test_image_ids = [], [], []

    for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(test_loader):

        images = images.to(device)
        reports_ids = reports_ids.to(device)
        B = images.size(0)


        batch_lambda = torch.full(
            (B,),
            float(current_lambda),
            dtype=torch.float32,
            device=device
        )


        output, _ = model(
            images,
            parameter_lambda=batch_lambda,
            mode='sample'
        )


        reports = tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

        test_res.extend(reports)
        test_gts.extend(ground_truths)

        if torch.is_tensor(images_id):
            images_id = images_id.cpu().tolist()
        test_image_ids.extend(images_id)


    os.makedirs(save_dir, exist_ok=True)

    actual_xlsx_name = f"lambda_{current_lambda}_{xlsx_name}"
    excel_path = os.path.join(save_dir, actual_xlsx_name)

    df = pd.DataFrame({
        "image_id": test_image_ids,
        "ground_truth": test_gts,
        "prediction": test_res,
    })
    df.to_excel(excel_path, index=False)




    test_met = compute_scores(
        {i: [gt] for i, gt in enumerate(test_gts)},
        {i: [re] for i, re in enumerate(test_res)},
        use_clinical=True
    )

    return test_met


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = prepare_device(n_gpu_use=1)

    tokenizer = Tokenizer(args)
    test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    model = R2GenModel(args, tokenizer).to(device)

    if args.resume and os.path.exists(args.resume):
        load_checkpoint(model, args.resume, device)
    else:
        raise FileNotFoundError(f'Checkpoint not found: {args.resume}')


    lambda_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    all_results = []

    for lmbda in lambda_list:

        
        metrics = test_only(model, tokenizer, test_loader, device, args.save_dir, args.xlsx_name, lmbda)
        

        metrics['lambda'] = lmbda
        all_results.append(metrics)




    metrics_df = pd.DataFrame(all_results)
    metrics_save_path = os.path.join(args.save_dir, 'all_lambda_metrics_summary.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
    



if __name__ == '__main__':
    main()
