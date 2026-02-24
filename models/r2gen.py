import torch
import torch.nn as nn
import numpy as np
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        self.param_encoder = None
        self.param_scale = nn.Parameter(torch.ones(1) * 0.1)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def _ensure_param_tensor(self, parameter_lambda, device):
        if parameter_lambda is None:
            return None
        if not torch.is_tensor(parameter_lambda):
            parameter_lambda = torch.tensor(parameter_lambda, dtype=torch.float32, device=device)
        else:
            parameter_lambda = parameter_lambda.to(device=device, dtype=torch.float32)

        if parameter_lambda.dim() == 0:
            parameter_lambda = parameter_lambda.view(1, 1)
        elif parameter_lambda.dim() == 1:
            parameter_lambda = parameter_lambda.unsqueeze(1)
        return parameter_lambda

    def _build_param_encoder_if_needed(self, fc_dim, device):
        if self.param_encoder is None:
            self.param_encoder = nn.Sequential(
                nn.Linear(1, fc_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(fc_dim), 
                nn.Linear(fc_dim, fc_dim),
            ).to(device)

            for m in self.param_encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward_mimic_cxr(self, images, parameter_lambda=None, targets=None, mode='train', update_opts={}):

        att_feats, fc_feats = self.visual_extractor(images)

        if parameter_lambda is not None:
            parameter_lambda = self._ensure_param_tensor(parameter_lambda, device=fc_feats.device)
            fc_dim = att_feats.size(2)
            self._build_param_encoder_if_needed(fc_dim=fc_dim, device=fc_feats.device)

            


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', lambd = parameter_lambda)
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, lambd = parameter_lambda)
            return output, output_probs
        else:
            raise ValueError(f"Unknown mode: {mode}")


    def forward_iu_xray(self, images, parameter_lambda=None, targets=None, mode='train', update_opts={}):

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if parameter_lambda is not None:

            parameter_lambda = self._ensure_param_tensor(parameter_lambda, device=fc_feats.device)

            visual_dim = att_feats.size(2) 
            self._build_param_encoder_if_needed(fc_dim=visual_dim, device=fc_feats.device)
            


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', lambd = parameter_lambda)
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, lambd = parameter_lambda)
            return output, output_probs