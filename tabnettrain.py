##tabnet classifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

import pandas as pd
import numpy as np
np.random.seed(0)

import os

from pytorch_tabnet.pretraining import TabNetPretrainer

def TabNetTest(tr,trl):
    index = np.random.choice(["train", "valid"], p =[.90, .1], size=(tr.shape[0],))
    X_train = tr[index=="train"]
    y_train = trl[index=="train"]
    
    X_valid = tr[index=="valid"]
    y_valid = trl[index=="valid"]
    
    unsupervised_model = TabNetPretrainer(
        cat_idxs=[],
        cat_dims=[],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        mask_type='entmax' # "sparsemax"
        )
    
    max_epochs = 1000 if not os.getenv("CI", False) else 2

    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        max_epochs=max_epochs , patience=5,
        batch_size=2048, virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8,  
        )
    
    clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":10, # how to use learning rate scheduler
                                         "gamma":0.90},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='sparsemax' # This will be overwritten if using pretrain model
                      )
    
    clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=max_epochs , patience=20,
    batch_size=1024, virtual_batch_size=32,
    num_workers=0,
    weights=1,
    drop_last=False,
    from_unsupervised=unsupervised_model
    )
    
    return clf