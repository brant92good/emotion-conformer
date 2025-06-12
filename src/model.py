import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model


class EmotionWav2vec2Conformer(nn.Module):
    def __init__(self, checkpoint_path: str, hidden_dim: int, num_classes: int, freeze_frontend: bool = True, device: str = None):
        super(EmotionWav2vec2Conformer, self).__init__()
        self.device = device
        self.frontend_model = self._load_pretrained_model(checkpoint_path)
        self.freeze_frontend = freeze_frontend
        
        if self.freeze_frontend:
            for param in self.frontend_model.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.frontend_model.encoder.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _load_pretrained_model(self, checkpoint_path: str):
        model, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path],
        )
        model: Wav2Vec2Model = model[0]
        model.eval()
        return model
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None, return_hidden_states: bool = False):
        x = x.to(self.device)
        padding_mask = padding_mask.to(self.device) if padding_mask is not None else None
        
        if self.freeze_frontend:
            with torch.no_grad():
                features = self.frontend_model.extract_features(x, padding_mask=padding_mask)
        else:
            features = self.frontend_model.extract_features(x, padding_mask=padding_mask)
        
        x = features['x']  # (B, T', D)
        
        # filter out padding
        if 'padding_mask' in features and features['padding_mask'] is not None:
            _padding_mask: torch.Tensor = features['padding_mask']
            mask = (~_padding_mask).unsqueeze(-1).float()  # (B, T', 1)
            x = x * mask
            x = x.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        
        x = self.classifier(x)
        if return_hidden_states:
            hidden_state = features['x']
            return x, hidden_state
        else:
            return x
