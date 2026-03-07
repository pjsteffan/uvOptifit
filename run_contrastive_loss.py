from tqdm import tqdm
import h5py
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch
import lightning as L
import torch
import torch.nn as nn

from collections import Counter

class WRsmallepoch(Dataset):
    def __init__(self, data_file: str, annotation_file: str, sample_rate: int = 5000):
        self.data_file = data_file
        self.annotation_file = annotation_file
        self.annotations = self.load_annotations()
        self.sample_rate = sample_rate
        self.frequencies = self.compute_frequency_vector()
        self.freq_weights = torch.Tensor(np.roll(np.unique(self.frequencies),1))

    def compute_frequency_vector(self):
        # Example vector
        epochs = self.annotations['epoch_id'].to_list()
        # Step 1: Count occurrences of each number
        counts = Counter(epochs)

        # Step 2: Calculate relative frequency
        total_count = len(epochs)
        relative_frequency = {num: count / total_count for num, count in counts.items()}

        # Step 3: Replace each number with its relative frequency
        result_vector = [relative_frequency[num] for num in epochs]
        return torch.Tensor(result_vector)
    
    def load_annotations(self):
        annotations = pd.read_csv(self.annotation_file)


        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        start_time = annotation['start_time']
        end_time = annotation['stop_time']
        label = annotation['epoch_id']

        start_index = int(start_time * self.sample_rate)
        #stop_index = int(end_time * self.sample_rate)
        with h5py.File(self.data_file, 'r') as f:
            ch1_data = f['Ch.1'][start_index:start_index+7500]
            ch2_data = f['Ch.2'][start_index:start_index+7500]
        
        ch1_data = self.downsample(ch1_data, original_fs=self.sample_rate, target_fs=100)
        ch2_data = self.downsample(ch2_data, original_fs=self.sample_rate, target_fs=100)

        ch1_data = self.filter_data(ch1_data, lowcut=5, highcut=30, fs=100.0, order=5)
        ch2_data = self.filter_data(ch2_data, lowcut=5, highcut=30, fs=100.0, order=5)

        _, ch1_data = self.power_spectrum(ch1_data, fs=100.0)
        _, ch2_data = self.power_spectrum(ch2_data, fs=100.0)

        
        epoch_data = np.stack([ch1_data, ch2_data], axis=0)  # Shape: (2, num_samples)
        epoch_data = epoch_data.transpose(1, 0)  # Shape: (num_samples, 2)

        return (epoch_data, label)

    def downsample(self, data, original_fs=5000, target_fs=100):
        """
        Downsample the data to the target frequency using 1D interpolation.

        Parameters:
        - data: The original data array.
        - original_fs: The original sampling frequency (default is 5000 Hz).
        - target_fs: The target sampling frequency (default is 100 Hz).

        Returns:
        - downsampled_data: The data resampled to the target frequency.
        """
        duration = len(data) / original_fs
        time_original = np.linspace(0, duration, len(data))
        time_target = np.linspace(0, duration, int(duration * target_fs))

        interpolator = interp1d(time_original, data, kind='linear')
        downsampled_data = interpolator(time_target)

        return downsampled_data
    
    def filter_data(self, data, lowcut=5, highcut=30, fs=100.0, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def power_spectrum(self, data, fs=100.0):
        freqs, psd = welch(data, fs)
        return freqs, np.log1p(psd) 


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, weight=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.nan_to_num(loss, nan=0.0)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class SupConFC(L.LightningModule):
    def __init__(self, input_size: int, num_layers: int, hidden_size: int, output_size: int,weight=None):
        super(SupConFC, self).__init__()
        self.save_hyperparameters()
        #self.encoder = Encoder(input_size, hidden_size, output_size, num_layers)
        # weight is currently unused in SupConLoss; sampler handles imbalance
        self.criterion = SupConLoss()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, int(input_size/2))
        self.fc3 = nn.Linear(int(input_size/2), hidden_size)
        self.activation = nn.ReLU()
        # optional stabilization on embeddings
        self.emb_norm = nn.LayerNorm(hidden_size)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Encode the source sequence
        out1 = self.fc1(source)
        out2 = self.fc2(self.activation(out1))
        hidden = self.fc3(self.activation(out2))
        hidden = self.emb_norm(hidden)

        return hidden
    

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, labels = batch

        # move to device with correct dtypes
        data = data.to(self.device).float()           # [B, F, C] where C=2
        labels = labels.to(self.device).long().view(-1)  # [B]

        # select single view/features (channel 0), expected to be [B, input_size]
        x = data[:, :, 0]

        # per-sample normalization (safer for contrastive learning than batchwise)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        # forward to embeddings [B, D]
        embeddings = self(x)
        # SupCon expects [B, n_views, D]; we have 1 view
        embeddings = embeddings.unsqueeze(1)

        # compute supervised contrastive loss
        loss = self.criterion(embeddings, labels=labels)

        # diagnostics (optional): embedding norm to detect collapse
        with torch.no_grad():
            emb_norm_mean = embeddings.norm(dim=-1).mean()
        self.log('train_loss', loss)
        self.log('emb_norm_mean', emb_norm_mean)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, labels = batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        output = self(data)
        loss = nn.functional.mse_loss(output, labels)
        self.log('test_loss', loss)
 

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def equal_var_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif '.gru.bias' in name:
            param.data.fill_(0)
        else:
            # Handle both 2D weight matrices and 1D parameters (e.g., LayerNorm weights)
            if param.data.dim() >= 2:
                fan_in = param.shape[1]
                param.data.normal_(std=1.0 / math.sqrt(fan_in))
            else:
                # For vectors (e.g., LayerNorm weights), use a small std
                param.data.normal_(std=1e-2)



if __name__ == "__main__":


    dataset = WRsmallepoch(
        data_file = '/app/Data/WR/WR5_Run4.hdf5', 
        annotation_file = '/app/Data/WR/Annotations/CSV_260206_annotations.csv', 
        sample_rate= 5000
    )

    datasampler = WeightedRandomSampler(weights=1/dataset.frequencies, num_samples=len(dataset), replacement=True)
    
    #batched_datasampler = BatchSampler(datasampler, batch_size=500, drop_last=True)

    loader = DataLoader(dataset, batch_size=100, sampler=datasampler)


    sup = SupConFC(input_size=76, num_layers = 4, hidden_size=32, output_size=16, weight=dataset.freq_weights)

    equal_var_init(sup)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(max_epochs=200,log_every_n_steps=15, accelerator="gpu", devices=1)
    trainer.fit(model=sup, train_dataloaders=loader)
