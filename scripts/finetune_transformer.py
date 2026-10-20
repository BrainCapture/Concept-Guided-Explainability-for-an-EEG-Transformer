from jsonargparse import CLI
import torch
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from typing import Optional

from cgx.transformer.linearheadbendr import LinearHeadBENDR
from cgx.eegprep_compat import StandardizeLabel, H5PYDatasetLabeled

import numpy as np
from collections import Counter


def get_class_counts(dataset):
    class_counts = Counter()

    for i, (d, l) in enumerate(dataset):
        class_counts[l.item()] += 1

    return class_counts

def use_dataset_path(dataset_path, logger, batch_size, num_workers, downsample_majority, channel_names, num_channels, seed):
    
    # Transformer and Dataset
    transformer = StandardizeLabel()
    dataset = H5PYDatasetLabeled(dataset_path, transform=transformer)

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    if channel_names is not None:
        if num_channels is not None:
            dataset.set_channel_subset(num_channels=num_channels)
        else:
            dataset.set_channel_subset(selected_channels=channel_names)

    if downsample_majority:
        print("Downsampling majority class")

        dataset.downsample_majority_class()

        index_train_val_temp, index_test_temp = next(group_splitter.split(X=torch.zeros(len(dataset.balanced_indices)), groups=dataset.subjects[dataset.balanced_indices]))
        index_train_temp, index_val_temp = next(group_splitter.split(X=torch.zeros(len(index_train_val_temp)), groups=dataset.subjects[dataset.balanced_indices][index_train_val_temp]))

        index_train = dataset.balanced_indices[index_train_temp]
        index_val = dataset.balanced_indices[index_val_temp]
        index_test = dataset.balanced_indices[index_test_temp]

    else:
        index_train_val, index_test = next(group_splitter.split(X=torch.zeros(len(dataset)), groups=dataset.subjects))
        index_train, index_val = next(group_splitter.split(X=torch.zeros(len(index_train_val)), groups=dataset.subjects[index_train_val]))

    data_train = torch.utils.data.Subset(dataset, index_train)
    data_val = torch.utils.data.Subset(dataset, index_val)
    data_test = torch.utils.data.Subset(dataset, index_test)

    #logger.log_hyperparams({'train_class_counts': get_class_counts(data_train), 
    #                        'val_class_counts': get_class_counts(data_val), 
    #                        'test_class_counts': get_class_counts(data_test)})

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    return dataloader_train, dataloader_val, dataloader_test

    

def use_dataset_train_and_eval_paths(dataset_train_path, dataset_eval_path, logger, batch_size, num_workers, downsample_majority, seed):

    # Transformer and Dataset
    transformer = StandardizeLabel()
    dataset_train = H5PYDatasetLabeled(dataset_train_path, transform=transformer)
    dataset_eval = H5PYDatasetLabeled(dataset_eval_path, transform=transformer)

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    if downsample_majority:
        print("Downsampling majority class")

        dataset_train.downsample_majority_class()

        index_train_temp, index_val_temp = next(group_splitter.split(X=torch.zeros(len(dataset_train.balanced_indices)), groups=dataset_train.subjects[dataset_train.balanced_indices]))

        index_train = dataset_train.balanced_indices[index_train_temp]
        index_val = dataset_train.balanced_indices[index_val_temp]

    else:
        index_train, index_val = next(group_splitter.split(X=torch.zeros(len(dataset_train)), groups=dataset_train.subjects))

    data_train = torch.utils.data.Subset(dataset_train, index_train)
    data_val = torch.utils.data.Subset(dataset_train, index_val)

    #logger.log_hyperparams({'train_class_counts': get_class_counts(data_train), 
    #                        'val_class_counts': get_class_counts(data_val), 
    #                        'test_class_counts': get_class_counts(dataset_eval)})

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test


def finetune(encoder_path: Optional[str] = None, name: str = 'model_1',
             dataset_path: Optional[str] = None, 
             dataset_train_path: Optional[str] = None, dataset_eval_path: Optional[str] = None,
             batch_size: int = 16, num_workers: int = 8, num_epochs: int = 500, seed: int = 42, 
             device: str = 'cuda', n_device: int = 1,in_features: int = 19, out_features: int = 2, 
             freeze_encoder: bool = False, patience: int = 100, encoder_h: int = 512,
             downsample_majority: bool = False, upsample_minority: bool = False, 
             channel_names: Optional[list] = None, num_channels: Optional[int] = None,
             project_name: str = 'finetune_experiments', use_wandb: bool = False,
             out_dir: str = 'model_weights/finetuned'):
        
    # Set all seeds
    seed_everything(seed, workers=True)

    if use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name=name, project=project_name)
    else:
        logger = CSVLogger(save_dir=out_dir, name=name)

    # Ensure arguments are mutually exclusive and required
    if dataset_path is None and (dataset_train_path is None or dataset_eval_path is None):
        raise ValueError(
            "You must provide either 'dataset_path' or both 'dataset_train_path' and 'dataset_eval_path'."
        )
    if dataset_path is not None and (dataset_train_path is not None or dataset_eval_path is not None):
        raise ValueError(
            "Provide either 'dataset_path' or both 'dataset_train_path' and 'dataset_eval_path', but not all three."
        )

    if dataset_path:
        print(f"Using dataset_path: {dataset_path}")

        dataloader_train, dataloader_val, dataloader_test = use_dataset_path(dataset_path, logger, batch_size, num_workers, 
                                                                             downsample_majority, channel_names, num_channels, seed)

    else:
        print(f"Using dataset_train_path: {dataset_train_path} and dataset_eval_path: {dataset_eval_path}")

        dataloader_train, dataloader_val, dataloader_test = use_dataset_train_and_eval_paths(dataset_train_path, dataset_eval_path, logger, 
                                                                                             batch_size, num_workers, downsample_majority, seed)
        
    model = LinearHeadBENDR(encoder_path, encoder_h=encoder_h, in_features=in_features, out_features=out_features)
    #model = FullBENDR(encoder_path, encoder_h=encoder_h, in_features=in_features, out_features=out_features)
    
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    logger.log_hyperparams({'batch_size': batch_size, 'num_workers': num_workers, 'num_epochs': num_epochs,
                            'seed': seed, 'device': device, 'n_device': n_device, 'out_features': out_features,
                            'freeze_encoder': freeze_encoder})

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    folder_name = out_dir

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_weights_only=True, dirpath=folder_name, filename=name, save_top_k=1)
    
    trainer = Trainer(accelerator='auto', devices=n_device, max_epochs=num_epochs,
                        callbacks=[model_checkpoint, early_stopping], logger=logger,
                        check_val_every_n_epoch=1, log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model, dataloaders=dataloader_test, ckpt_path='best')
    

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    CLI(finetune)