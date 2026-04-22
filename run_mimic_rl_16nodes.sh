#!/bin/bash
#SBATCH --job-name=mimic_16nodes_group
#SBATCH --account=PAS3128
#SBATCH --time=120:00:00        # 8节点跑得快，但MIMIC数据量大，建议给足时间
#SBATCH --nodes=16              # 申请8个节点
#SBATCH --ntasks=16             # 每个节点1个任务
#SBATCH --cpus-per-task=32     # 增加CPU核心数以加速数据读取
#SBATCH --gpus-per-node=2      # 每个节点2块显卡，总共16块
#SBATCH --partition=nextgen
#SBATCH --output=slurm_16nodesgroup_condition_8_2-%j.out

# 加载环境
module load miniconda3/24.1.2-py310 cuda/12.8.1
source activate /users/PAS3128/lingchen/miniconda3/envs/r2gen_torch171

# 1. 自动定位主节点
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# 获取主节点的内网数字 IP (最稳定)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w $MASTER_ADDR hostname -I | awk '{print $1}')

# 2. 针对 8 节点的 NCCL 网络优化
export NCCL_SOCKET_IFNAME=em3     # 强制走 em3 网卡
export GLOO_SOCKET_IFNAME=em3
export NCCL_IB_DISABLE=1         # 如果没有 InfiniBand，必须禁用
export NCCL_P2P_DISABLE=1        # 跨节点通常不支持 P2P
export NCCL_DEBUG=INFO           # 开启日志，方便报错时排查网络
export TORCH_DISTRIBUTED_DEBUG=DETAIL # 开启分布式调试模式

# 3. 环境变量防止超时
export NCCL_BLOCKING_WAIT=1 
export NCCL_TIMEOUT=3600         # 增加超时时间到1小时，防止大规模初始化时卡死

echo "Master Node: $MASTER_ADDR (IP: $MASTER_IP)"
echo "Total Nodes: $SLURM_NNODES"

# 4. 启动分布式训练
# --nnodes=8: 告诉 torchrun 共有8个节点
# --nproc_per_node=2: 每个节点启动2个进程（对应2块显卡）
srun torchrun \
    --nnodes=16 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:25678 \
    /fs/ess/PAS3128/lingchen/code/LHR_RFL/LHR-RFL-main2/train_run3_mimic_ddp_16nodes_final.py
    --image_dir /fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/files/ \
    --ann_path /fs/ess/PAS3128/lingchen/data/MIMIC-CXR-JPG/mimic-cxr-jpg-2.0.0.physionet.org/mimic_annotation.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 4 \
    --epochs 20 \
    --save_dir results_MIMIC16nodes_group2_clinical01_8_2/MIMIC-CXR \
    --record_dir results_MIMIC16nodes_group2_clinical01_8_2 \
    --bleu_weight 5/11 \
    --meteor_weight 1/11 \
    --rouge_weight 5/11 \
    --clinical_weight 0.5 \
    --train_sample_n 5 \

