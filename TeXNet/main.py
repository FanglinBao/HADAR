import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

import torch
import numpy as np
import os
import json

from config import parse_args
from model import SMPModel
# from datamodule import HADARLoader
from datamodule_newdata import HADARMultipleScenesLoader

if __name__ == "__main__":
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    model_checkpoint = True
    if args.checkpoint_dir == '' or args.checkpoint_dir is None:
        args.checkpoint_dir = 'tmp_ckpt'
        model_checkpoint = False

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if model_checkpoint:
        logger = TensorBoardLogger(save_dir=args.checkpoint_dir,
                                   version=1,
                                   name='lightning_logs')
    else:
        logger = None
    
    overfit_batches = 0
    if args.overfit:
        overfit_batches = 2
    
    # AMP is Automatic Mixed Precision.
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    if args.use_amp:
        precision = 16 # bits
    else:
        precision = 32
    
    callback_list = []

    if model_checkpoint:
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, save_last=True)
        callback_list.append(checkpoint_callback)

    plugins_list = []
    
    model = SMPModel(args)
    # datamodule = HADARLoader(args)
    datamodule = HADARMultipleScenesLoader(args)

    if args.ngpus <= 1:
        sync_bn = False
    else:
        sync_bn = True
    
    if args.swa:
        from pytorch_lightning.callbacks import StochasticWeightAveraging as SWA
        swa = SWA(swa_lrs=1e-3)
        callback_list.append(swa)

    trainer = pl.Trainer(devices=args.ngpus,
                         strategy='ddp_find_unused_parameters_false',
                         accelerator="gpu",
                         plugins=plugins_list,
                         num_nodes=args.num_nodes,
                         amp_backend='native',
                         auto_lr_find=True,
                         benchmark=True, # only if the input sizes don't change rapidly
                         callbacks=callback_list,
                         default_root_dir=args.checkpoint_dir,
                         fast_dev_run=args.quick_check,
                         gradient_clip_val=args.grad_clip, # 0 gradient clip means no clipping
                         logger=logger,
                         check_val_every_n_epoch=args.eval_every,
                         max_epochs=args.epochs,
                         overfit_batches=overfit_batches,
                         precision=precision, # decides the use of AMP
                         sync_batchnorm=sync_bn,
                         detect_anomaly=True,
                         enable_progress_bar=True
                         )
    
    # dump training options to a JSON file.
    if model_checkpoint:
        json_dump_dict = vars(args)
        with open(os.path.join(args.checkpoint_dir, 'training_args.json'), 'w') as json_f:
            json.dump(json_dump_dict, json_f)
        print(f"** Dumped training arguments to {args.checkpoint_dir}/training_args.json")

    if args.eval:
        model = SMPModel.load_from_checkpoint(args.resume, args=args)
        trainer.validate(model, datamodule=datamodule)
    else:
        if args.resume != "":
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
        else:
            trainer.fit(model, datamodule=datamodule)
