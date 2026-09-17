import torch

from src._runtime_guard import (
    require_choice,
    require_divisible,
    require_int,
    require_probability,
)

class Para():
    # hyperparameters
    def __init__(
                 self,
                 pretrain_lr           = 1e-4,
                 finetune_lr           = 5e-5,
                 pretrain_epoch        = 100,
                 finetune_epoch        = 30,
                 pretrain_batch_size   = 64,
                 finetune_batch_size   = 32,
                 combination           = 20,
                 num_statis            = 8,
                 encoder_h_dim         = 768,
                 encoder_heads         = 8,
                 encoder_stalk_dim     = 96,
                 encoder_num_layers    = 5,
                 decoder_h_dim         = 512,
                 decoder_heads         = 8,
                 decoder_stalk_dim     = 64,
                 decoder_num_layers    = 3,
                 max_len               = 100+1,
                 low_rank              = 8,
                 encoder_dropout       = 0.1,
                 decoder_dropout       = 0.1,
                 mask_ratio            = 0.5,
                 mask_typ              = 'span', # random, span
                 norm_typ              = 'post_norm', # pre_norm, post_norm
                 patch_size            = 1,
                 weight_decay          = 0.05,
                 warmup_rate           = 0.05,
                 min_lr_ratio          = 0.01,
                 device                = torch.device('cuda'),
                 pretrain_seed         = 42,
                 scaler_folder         = './scaler-save',
                 model_folder          = './checkpoint',
                 model_name            = 'model-99.pt',
                 pdb_folder            = './data/all-pdbs',     # raw protein-liagnd complex 
                 pdb_feature_folder    = './data/euclidean-feature', # generated persistent laplacian
                 pqr_folder            = './data/all-pqrs',          # raw pqr file
                 pqr_feature_folder    = './data/charge-feature', # generated charge-based persistent laplacian
                 seeds                 = [42,1,2,3,4,5,6,7,8,9],
                 save_folder           = './prediction'
                 ):
                 
        self.pretrain_lr         = pretrain_lr
        self.finetune_lr         = finetune_lr
        self.pretrain_epoch      = pretrain_epoch
        self.finetune_epoch      = finetune_epoch
        self.pretrain_batch_size = pretrain_batch_size
        self.finetune_batch_size = finetune_batch_size
        self.combination         = combination
        self.num_statis          = num_statis
        self.encoder_h_dim       = encoder_h_dim
        self.encoder_heads       = encoder_heads
        self.encoder_stalk_dim   = encoder_stalk_dim
        self.encoder_num_layers  = encoder_num_layers
        self.decoder_h_dim       = decoder_h_dim
        self.decoder_heads       = decoder_heads
        self.decoder_stalk_dim   = decoder_stalk_dim
        self.decoder_num_layers  = decoder_num_layers
        self.max_len             = max_len
        self.low_rank            = low_rank
        self.encoder_dropout     = encoder_dropout
        self.decoder_dropout     = decoder_dropout
        self.mask_ratio          = mask_ratio
        self.mask_typ            = mask_typ
        self.norm_typ            = norm_typ
        self.patch_size          = patch_size
        self.weight_decay        = weight_decay
        self.warmup_rate         = warmup_rate
        self.min_lr_ratio        = min_lr_ratio
        self.device              = device
        self.pretrain_seed       = pretrain_seed
        self.scaler_folder       = scaler_folder
        self.model_folder        = model_folder
        self.model_name          = model_name
        self.pdb_folder          = pdb_folder
        self.pdb_feature_folder  = pdb_feature_folder
        self.pqr_folder          = pqr_folder
        self.pqr_feature_folder  = pqr_feature_folder
        self.seeds               = seeds
        self.save_folder         = save_folder
        self._verify_contract()
        
    def _verify_contract(self):
        integer_fields = (
            'pretrain_epoch', 'finetune_epoch', 'pretrain_batch_size',
            'finetune_batch_size', 'combination', 'num_statis',
            'encoder_h_dim', 'encoder_heads', 'encoder_stalk_dim',
            'encoder_num_layers', 'decoder_h_dim', 'decoder_heads',
            'decoder_stalk_dim', 'decoder_num_layers', 'max_len',
            'low_rank', 'patch_size',
        )
        for field in integer_fields:
            require_int(getattr(self, field), field, minimum=1)
        require_divisible(self.encoder_h_dim, self.encoder_heads, 'encoder dimension')
        require_divisible(self.decoder_h_dim, self.decoder_heads, 'decoder dimension')
        require_probability(self.encoder_dropout, 'encoder dropout')
        require_probability(self.decoder_dropout, 'decoder dropout')
        require_probability(self.mask_ratio, 'mask ratio')
        require_probability(self.warmup_rate, 'warmup rate')
        require_probability(self.min_lr_ratio, 'minimum learning-rate ratio')
        require_choice(self.mask_typ, ('random', 'span'), 'mask strategy')
        require_choice(self.norm_typ, ('pre_norm', 'post_norm'), 'normalization strategy')
        require_int(self.pretrain_seed, 'pretrain seed', minimum=0)
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ValueError('seeds must be a non-empty list or tuple')
        for seed in self.seeds:
            require_int(seed, 'fine-tune seed', minimum=0)
        if self.pretrain_lr <= 0 or self.finetune_lr <= 0:
            raise ValueError('learning rates must be positive')
        if self.weight_decay < 0:
            raise ValueError('weight decay cannot be negative')
        
    def print_attrs(self):
       d = self.__dict__
       w = max((len(k) for k in d), default=0)
       for k, v in d.items():
           print(f"{k:<{w}} : {v}")
        