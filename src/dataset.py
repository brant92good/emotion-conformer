import logging
from math import e
from pathlib import Path

import numpy as np
import regex
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import logger


class EmoType:
    '''
    Emotion type for dataset.
    '''
    class IEMOCAP:
        ANGRY = 'ang'
        HAPPY = 'hap'
        SAD = 'sad'
        NEUTRAL = 'neu'
        FRUSTRATED = 'fru'
        EXCITED = 'exc'
        FEARFUL = 'fea'
        SURPRISE = 'sur'
        DISGUST = 'dis'
        OTHER = 'xxx'
        ALL = [ANGRY, HAPPY, SAD, NEUTRAL, FRUSTRATED, EXCITED, FEARFUL, SURPRISE, DISGUST, OTHER]


class DatasetPreprocessor:
    def __init__(self):
        '''
        Prepare the dataset for training.
        '''
        self.logger = logging.getLogger('DatasetPreprocessor')
        self.logger.info('Initialized')
    
    def extract_iemocap_feature(self, d_path: str, emo_type: list = EmoType.IEMOCAP.ALL):
        '''
        Extract features from LibriSpeech dataset.
        Args:
            d_path: path to the dataset
            emo_type: list of emotion types to extract
        '''
        
        buf = {}
        skip = 0
        
        self.logger.info(f'Extracting features from {d_path}')
        self.logger.info(f'emo_type: {emo_type}')
        
        for index in range(1, 6):
            this_session_wav_dir = f'{d_path}/Session{index}/sentences/wav'
            
            for ses in Path(this_session_wav_dir).iterdir():
                if not ses.is_dir():
                    continue
                
                label_buffer = ''
                with Path(f'{d_path}/Session{index}/dialog/EmoEvaluation/{ses.name}.txt').open('r') as f:
                    label_buffer = f.read()
                
                for wav in ses.iterdir():
                    if not wav.is_file() or wav.suffix != '.wav':
                        skip += 1
                        continue
                    
                    waveforms = self.read_audio(wav)
                    if len(waveforms) < 16000:
                        skip += 1
                        continue
                    
                    # regex: Ses01F_impro01_F000	neu
                    reg = regex.search(rf'{wav.stem}\s+(\w+)', label_buffer)
                    if reg is None:
                        skip += 1
                        continue
                    wav_label = reg.group(1)
                    
                    if wav_label not in emo_type:
                        skip += 1
                        continue
                    
                    buf[wav.stem] = {
                        'waveforms': waveforms,
                        'label': wav_label
                    }
        
        # check waveforms
        waveform_avg = np.mean([len(v['waveforms']) for v in buf.values()])
        for k, v in buf.items():
            if len(v['waveforms']) > waveform_avg * 2:
                skip += 1
                continue
        
        self.logger.info(f'Extracted {len(buf)} samples, skipped {skip} samples.')
        return buf
    
    def read_audio(self, f_path):
        '''
        Read audio file and return the waveform.
        '''
        wav, sr = sf.read(f_path)
        channel = sf.info(f_path).channels
        assert sr == 16000, f'Sample rate should be 16kHz, but got {sr} in file {f_path}'
        assert channel == 1, f'Channel should be 1, but got {channel} in file {f_path}'
        
        return wav
    
    def pack_cache(self, buf: dict, emo_type: list, output_path: str):
        '''
        Pack the dataset into a .npy file.
        Args:
            output_path: path to the output file
        '''
        feature_path = Path(output_path + '/feature.npy')
        label_idx_path = Path(output_path + '/label_map.npy')
        
        
        np.save(feature_path, buf, allow_pickle=True)
        size = Path(feature_path).stat().st_size
        self.logger.info(f'Packing {len(buf)} samples into {feature_path} done')
        self.logger.info(f'File size: {size / 1024 / 1024 / 1024:.2f} GB')
        
        # label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(emo_type)}
        np.save(label_idx_path, label_to_idx, allow_pickle=True)
        self.logger.info(f'label index: {label_to_idx}')
        self.logger.info(f'Packing label index mapping into {label_idx_path} done')


class IEMOCAP_Dataset(Dataset):
    def __init__(self, npy_path: str, label_map: dict):
        '''
        Initialize the dataset.
        Args:
            dataset_path: path to the dataset
        '''
        dataset_npy: np.ndarray = np.load(npy_path, allow_pickle=True)
        self.dataset: dict = dataset_npy.item()
        self.keys = list(self.dataset.keys())
        
        label_map_npy: np.ndarray = np.load(label_map, allow_pickle=True)
        self.label_map: dict = label_map_npy.item()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.dataset[key]
        waveform = item['waveforms']
        label = item['label']
        return {
            'waveform': waveform,
            'label': self.label_map[label]
        }

class ASVspoof2019_Dataset(Dataset):
    def __init__(self, metadata_path: str, audio_folder: str, audio_extension: str = '.flac'):
        '''
        Initialize the dataset.
        Args:
            metadata_path: path to the metadata file
            audio_folder: path to the folder containing audio files
            audio_extension: audio file extension (default: '.flac')
        '''
        self.audio_folder = audio_folder
        self.audio_extension = audio_extension
        self.data = []
        
        # Create label map (same structure as IEMOCAP)
        self.label_map = {
            'bonafide': 0,
            'spoof': 1
        }
        
        # Read metadata file
        with open(metadata_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure we have enough columns
                    filename = parts[1]  # Second column
                    label = parts[-1]    # Last column
                    
                    # Create full audio path
                    audio_path = os.path.join(audio_folder, f"{filename}{audio_extension}")
                    
                    self.data.append({
                        'audio_path': audio_path,
                        'label': label,  # Keep original label string
                        'filename': filename
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio waveform
        waveform, sample_rate = torchaudio.load(item['audio_path'])
        
        return {
            'waveform': waveform,
            'label': self.label_map[item['label']]  # Map label using label_map
        }


class PadCollator:
    def __init__(self, mode='train', fixed_len=16000 * 4):
        assert mode in ['train', 'val', 'test'], f'Invalid mode {mode}'
        self.mode = mode
        self.fixed_len = fixed_len
    
    def __call__(self, batch):
        waveforms = [torch.tensor(item['waveform']) for item in batch]
        labels = [item['label'] for item in batch]
        
        pad_len = self.fixed_len
        
        padded_waveforms = []
        padding_masks = []
        
        for w in waveforms:
            length = w.shape[0]
            
            if length > pad_len:
                w = w[:pad_len]
                padding_mask = torch.zeros(pad_len, dtype=torch.bool)  # 全部有效
            else:
                padding_mask = torch.cat([
                    torch.zeros(length, dtype=torch.bool),          # 有效區段
                    torch.ones(pad_len - length, dtype=torch.bool)  # padding 區段
                ])
                w = torch.nn.functional.pad(w, (0, pad_len - length))
            
            padded_waveforms.append(w)
            padding_masks.append(padding_mask)
        
        waveform_tensor = torch.stack(padded_waveforms)        # (B, T)
        label_tensor = torch.tensor(labels)                    # (B,)
        padding_mask_tensor = torch.stack(padding_masks)       # (B, T)
        
        return {
            'waveform': waveform_tensor,
            'label': label_tensor,
            'padding_mask': padding_mask_tensor
        }


class IEMOCAP_DataLoader:
    def __init__(
            self,
            d_path: str,
            ratios: list = [0.8, 0.1, 0.1],
        ):
        '''
        Initialize the dataloader.
        Args:
            dataset_path: path to the dataset
            batch_size: batch size
        '''
        self.logger = logging.getLogger('IEMOCAP_DataLoader')
        self.logger.info('Initialized')
        
        npy_path = Path(d_path + '/feature.npy')
        assert npy_path.exists(), f'File {npy_path} does not exist'
        
        label_map_path = Path(d_path + '/label_map.npy')
        assert label_map_path.exists(), f'File {label_map_path} does not exist'
        
        self.logger.info(f'Loading dataset from {npy_path}')
        
        dataset = IEMOCAP_Dataset(npy_path, label_map_path)
        
        self.train_size = int(len(dataset) * ratios[0])
        self.val_size = int(len(dataset) * ratios[1])
        self.test_size = len(dataset) - self.train_size - self.val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [self.train_size, self.val_size, self.test_size]
        )
        self.train_dataset: Subset = train_dataset
        self.val_dataset: Subset = val_dataset
        self.test_dataset: Subset = test_dataset
    
    def get_dataloader(self, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        '''
        Get the dataloader.
        Args:
            batch_size: batch size
            shuffle: whether to shuffle the dataset
        '''
        
        train_collator = PadCollator(mode='train')
        val_collator = PadCollator(mode='val')
        test_collator = PadCollator(mode='test')
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=train_collator
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=val_collator
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=test_collator
        )
        return train_loader, val_loader, test_loader

class ASVspoof2019_DataLoader:
    def __init__(
        self,
        train_metadata_path: str,    # Path to train metadata file
        train_audio_folder: str,     # Path to train audio folder
        dev_metadata_path: str,      # Path to dev metadata file  
        dev_audio_folder: str,       # Path to dev audio folder
        audio_extension: str = '.flac',  # Audio file extension
        ratios: list = [0.8, 0.2],   # Only for train/val split
    ):
        '''
        Initialize the dataloader.
        Args:
            train_metadata_path: path to the training metadata file
            train_audio_folder: path to the training audio folder
            dev_metadata_path: path to the dev metadata file
            dev_audio_folder: path to the dev audio folder
            audio_extension: audio file extension (default: '.flac')
            ratios: [train_ratio, val_ratio] for splitting training data
        '''
        self.logger = logging.getLogger('ASVspoof_DataLoader')
        self.logger.info('Initialized')
        
        # Validate training dataset paths
        assert Path(train_metadata_path).exists(), f'File {train_metadata_path} does not exist'
        assert Path(train_audio_folder).exists(), f'Folder {train_audio_folder} does not exist'
        
        # Validate dev dataset paths
        assert Path(dev_metadata_path).exists(), f'File {dev_metadata_path} does not exist'
        assert Path(dev_audio_folder).exists(), f'Folder {dev_audio_folder} does not exist'
        
        self.logger.info(f'Loading training dataset from {train_metadata_path}')
        self.logger.info(f'Loading dev dataset from {dev_metadata_path}')
        
        # Load training dataset and split for train/val
        train_dataset = ASVspoof2019_Dataset(
            metadata_path=train_metadata_path,
            audio_folder=train_audio_folder,
            audio_extension=audio_extension
        )
        
        self.train_size = int(len(train_dataset) * ratios[0])
        self.val_size = len(train_dataset) - self.train_size
        
        train_split, val_split = random_split(
            train_dataset,
            [self.train_size, self.val_size]
        )
        
        # Load separate dev dataset for testing
        test_dataset = ASVspoof2019_Dataset(
            metadata_path=dev_metadata_path,
            audio_folder=dev_audio_folder,
            audio_extension=audio_extension
        )
        
        self.train_dataset: Subset = train_split
        self.val_dataset: Subset = val_split
        self.test_dataset: Dataset = test_dataset  # This uses the dev set
    
    def get_dataloader(self, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        '''
        Get the dataloader.
        Args:
            batch_size: batch size
            num_workers: number of worker processes
            shuffle: whether to shuffle the training dataset (val/test are never shuffled)
        '''
        
        train_collator = PadCollator(mode='train')
        val_collator = PadCollator(mode='val')
        test_collator = PadCollator(mode='test')
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # Only shuffle training data
            num_workers=num_workers,
            collate_fn=train_collator
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle validation
            num_workers=num_workers,
            collate_fn=val_collator
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle test
            num_workers=num_workers,
            collate_fn=test_collator
        )
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    IEMOCAP_DATASET_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release'
    
    OUTPUT_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release'
    
    SELECT_EMO_LIST = [EmoType.IEMOCAP.ANGRY, EmoType.IEMOCAP.HAPPY, EmoType.IEMOCAP.SAD, EmoType.IEMOCAP.NEUTRAL]
    
    dataset_preprocessor = DatasetPreprocessor()
    buf = dataset_preprocessor.extract_iemocap_feature(IEMOCAP_DATASET_PATH, SELECT_EMO_LIST)
    dataset_preprocessor.pack_cache(buf, SELECT_EMO_LIST, OUTPUT_PATH)
    
    iemocap_dataloader = IEMOCAP_DataLoader(OUTPUT_PATH)
    train_loader, val_loader, test_loader = iemocap_dataloader.get_dataloader(batch_size=32)
    for batch in train_loader:
        print(batch['waveform'].shape)
        print(batch['label'])
        break