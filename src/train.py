import argparse
import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from hydra.core.config_store import ConfigStore
from nvitop import CudaDevice, ResourceMetricCollector
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config import Config
from dataset import IEMOCAP_DataLoader, ASVspoof2019_DataLoader
from model import EmotionWav2vec2Conformer, ASVspoofWav2vec2Conformer


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger('Trainer')
        self.logger.info('Initializing Trainer')
        
        self.check_dir()
        self.check_device()
        
        self.collector = ResourceMetricCollector(devices=CudaDevice.all(),
                                                root_pids={os.getpid()},
                                                interval=1.0)
        
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
    
    def check_dir(self):
        '''
        Check if the directory exists.
        '''
        self.output_dir = os.getcwd()
        self.checkpoint_dir = Path(self.output_dir) / 'checkpoint'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = Path(self.output_dir) / 'tensorboard'
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Tensorboard cmd: tensorboard --logdir={self.tensorboard_dir} --host=0.0.0.0 --port=6006')
    
    def check_device(self):
        '''
        Check if the device is available.
        '''
        if torch.cuda.is_available():
            self.cfg.common.device = 'cuda'
            self.logger.info('Using GPU')
        else:
            self.cfg.common.device = 'cpu'
            self.logger.info('Using CPU')
    
    def write_tensorboard(self, res: dict, epoch: int):
        '''
        Write the result to tensorboard.
        '''
        for key, value in res.items():
            self.writer.add_scalar(key, value, epoch)
        
        # stats
        stats = self.collector.collect()
        selected_stats = [
            'resource/host/cpu_percent (%)/last',
            'resource/host/memory_percent (%)/last',
            'resource/host/memory_used (GiB)/last',
            'resource/cuda:0 (gpu:0)/memory_used (MiB)/last',
            'resource/cuda:0 (gpu:0)/memory_free (MiB)/last',
            'resource/cuda:0 (gpu:0)/memory_percent (%)/last',
            'resource/cuda:0 (gpu:0)/gpu_utilization (%)/last',
            'resource/cuda:0 (gpu:0)/temperature (C)/last',
            'resource/cuda:0 (gpu:0)/power_usage (W)/last',
        ]
        for stat in selected_stats:
            if stat in stats:
                self.writer.add_scalar(stat, stats[stat], epoch)
    
    def train(
            self,
            model: ASVspoofWav2vec2Conformer,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn,
            train_loader: ASVspoof2019_DataLoader
        ):
        '''
        Train the model.
        '''
        model.to(self.cfg.common.device)
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            waveforms: torch.Tensor = batch['waveform']
            waveforms = waveforms.to(self.cfg.common.device).float()
            padding_mask: torch.Tensor = batch['padding_mask']
            padding_mask = padding_mask.to(self.cfg.common.device)
            labels: torch.Tensor = batch['label']
            labels = labels.to(self.cfg.common.device)
            
            outputs = model(waveforms, padding_mask)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    @torch.no_grad()
    def val_test(
            self,
            model: ASVspoofWav2vec2Conformer,
            criterion: torch.nn.Module,
            val_loader: ASVspoof2019_DataLoader,
            test_loader: ASVspoof2019_DataLoader
        ):
        '''
        Validate and test the model.
        '''
        model.eval()
        # val, test
        loss = [0.0, 0.0]
        correct = [0, 0]
        total = [0, 0]
        
        for i, loader in enumerate([val_loader, test_loader]):
            for batch in loader:
                waveforms: torch.Tensor = batch['waveform']
                waveforms = waveforms.to(self.cfg.common.device).float()
                padding_mask: torch.Tensor = batch['padding_mask']
                padding_mask = padding_mask.to(self.cfg.common.device)
                labels: torch.Tensor = batch['label']
                labels = labels.to(self.cfg.common.device)
                
                outputs = model(waveforms, padding_mask)
                loss[i] += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total[i] += labels.size(0)
                correct[i] += (predicted == labels).sum().item()
        
        res = {
            'val_loss': loss[0] / len(val_loader),
            'val_acc': 100 * correct[0] / total[0],
            #'val_f1': 100 * (2 * correct[0] / (total[0] + correct[0])),
            'test_loss': loss[1] / len(test_loader),
            'test_acc': 100 * correct[1] / total[1],
            #'test_f1': 100 * (2 * correct[1] / (total[1] + correct[1]))
        }
        return res
    
    def display_score(self, res):
        '''
        Display the score of the model.
        '''
        table = [
            ['Set', 'Loss', 'Accuracy'],
            ['Train',
                f"{res['train_loss']:.4f}",
            ],
            ['Validation',
                f"{res['val_loss']:.4f}",
                f"{res['val_acc']:.2f}%"
            ],
            ['Test',
                f"{res['test_loss']:.4f}",
                f"{res['test_acc']:.2f}%"
            ]
        ]
        ascii_table = tabulate(table[1:], headers=table[0], tablefmt='grid')
        self.logger.info('\n' + ascii_table)
    
    def k_fold_train(self, model: ASVspoofWav2vec2Conformer):
        '''
        Train the model using k-fold cross-validation.
        '''
        torch.manual_seed(self.cfg.common.seed)
        torch.cuda.manual_seed(self.cfg.common.seed)
        
        train_loader, val_loader = ASVspoof2019_DataLoader(
            train_metadata_path=self.cfg.dataset.train_metadata_path,
            train_audio_folder=self.cfg.dataset.train_audio_folder,
            dev_metadata_path=self.cfg.dataset.dev_metadata_path,
            dev_audio_folder=self.cfg.dataset.dev_audio_folder,
            audio_extension=self.cfg.dataset.audio_extension,
            ratios=self.cfg.dataset.ratios,
        ).get_dataloader(
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers
        )

        
        self.logger.info(model)
        
        optimizer = None
        if self.cfg.optimizer._name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        scheduler = None
        if self.cfg.scheduler._name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.train.epoch,
                eta_min=self.cfg.scheduler.eta_min,
                last_epoch=self.cfg.scheduler.last_epoch
            )
        
        last_test_acc = 0.0
        test_acc_avg = 0.0
        #test_f1_avg = 0.0
        
        # start the resource metric collector
        self.collector.start('resource')
        
        fold_epoch_bar = tqdm(range(self.cfg.train.fold * self.cfg.train.epoch), desc='Training', unit='epoch')
        
        for fold in range(self.cfg.train.fold):
            for epoch in range(self.cfg.train.epoch):
                start_time = time.time()
                
                self.logger.info(f'Fold {fold + 1}/{self.cfg.train.fold} - Epoch {epoch + 1}/{self.cfg.train.epoch}')
                self.logger.info(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
                fold_epoch_bar.set_description(f'Fold {fold + 1}/{self.cfg.train.fold} - Epoch {epoch + 1}/{self.cfg.train.epoch}')
                
                train_loss = self.train(model, optimizer, criterion, train_loader)
                scheduler.step()
                
                res = {}
                res['train_loss'] = train_loss
                res.update(self.val_test(model, criterion, val_loader, test_loader))
                # self.display_score(res)
                
                formatted_res = {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in res.items()}
                self.logger.info(formatted_res)
                
                test_acc_avg += res['test_acc']
                #test_f1_avg += res['test_f1']
                
                # write to tensorboard
                self.write_tensorboard(res, epoch)
                
                # save
                if res['test_acc'] > last_test_acc:
                    torch.save(model.state_dict(), f'{self.checkpoint_dir}/fold_{fold + 1}-best_model.pt')
                    last_test_acc = res['test_acc']
                    self.logger.info(f'Model saved at fold {fold + 1}, epoch {epoch + 1}')
                
                end_time = time.time()
                self.logger.info(f'Epoch {epoch + 1} finished in {end_time - start_time:.2f} seconds')
                fold_epoch_bar.update(1)
        
        self.logger.info(f'Best test accuracy: {last_test_acc:.2f}%')


if __name__ == '__main__':
    CONFIG_NAME = 'asv19'
    
    with open(f'/home/brant/Projects/emotion-conformer/config/{CONFIG_NAME}.yaml', 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    pre_config = OmegaConf.create(yaml_cfg)
    
    label_map : dict = pre_config.dataset.lable_map
    if not label_map:
        raise ValueError("Label map is empty. Please provide a valid label map in the configuration.")
    idx_to_label = {v: k for k, v in label_map.items()}
    
    model = ASVspoofWav2vec2Conformer(
        checkpoint_path=pre_config.model.frontend_model.path,
        hidden_dim=pre_config.model.hidden_dim,
        freeze_frontend=pre_config.model.get('freeze_frontend', True),
        device=pre_config.common.device
    )
    
    cs = ConfigStore.instance()
    cs.store(name='config', node=Config)
    
    @hydra.main(config_path='../config', config_name=f'{CONFIG_NAME}.yaml')
    def main(cfg: Config):
        trainer = Trainer(cfg)
        trainer.k_fold_train(model)
    
    main()