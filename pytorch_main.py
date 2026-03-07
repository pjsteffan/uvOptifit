import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data

import lightning as L

from datasets import WRsmallepoch
from models import GRUClassifier, equal_var_init


def main(data_file, annotation_file, sample_rate=5000):
    
    dataset = WRsmallepoch(
        data_file = data_file, 
        annotation_file = annotation_file, 
        sample_rate= sample_rate
    )
    
    
    trv_set_size = int(len(dataset) * 0.8)
    #test_set_size = len(dataset) - trv_set_size

    trv_indices = list(range(trv_set_size))
    #test_indices = list(range(trv_set_size, len(dataset)))


    trv_set = data.Subset(dataset, trv_indices)
    #test_set = data.Subset(dataset, test_indices)
    
    # use 20% of training data for validation
    train_set_size = int(len(trv_set) * 0.8)
    valid_set_size = len(trv_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(trv_set, [train_set_size, valid_set_size], generator=seed)

    train_indices = train_set.indices
    train_weights = 1.0 / dataset.frequencies[train_indices]
    datasampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_set), replacement=True)


    train_loader = DataLoader(train_set, batch_size=500,sampler=datasampler)
    valid_loader = DataLoader(valid_set, batch_size=500)
    #test_loader = DataLoader(test_set, batch_size=500)
    

    GRUc = GRUClassifier(input_size=1, num_layers = 1, hidden_size=512, output_size=16, out_size1=8, out_size2=6, logit_size=3)
    equal_var_init(GRUc)

    trainer = L.Trainer(max_epochs=200,log_every_n_steps=10, accelerator="gpu", devices=1)
    trainer.fit(GRUc, train_loader, valid_loader)
    #trainer.test(GRUc, dataloaders=test_loader)


if __name__ == "__main__":
    main('/app/Data/WR/WR5_Run4.hdf5', '/app/Data/WR/Annotations/260218_annotations_a.pkl')

