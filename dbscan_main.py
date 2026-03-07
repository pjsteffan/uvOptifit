from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


from datasets import WRsmallepoch


def main(data_file, annotation_file, sample_rate=5000):
    dataset = WRsmallepoch(
    data_file = data_file, 
    annotation_file = annotation_file,
    epoch_size=5.0, 
    sample_rate= sample_rate
    )

    print(len(dataset.annotations))

    batch_size = 500

    dataloader = DataLoader(dataset,batch_size=batch_size)
    
    DATA = np.empty((len(dataset.annotations), 76))
    for i, batch in tqdm(enumerate(dataloader)):

        data, _ = batch
        data = data.numpy()

        DATA[i*batch_size:i*batch_size+batch_size,:] = data




    means = np.mean(DATA, axis=0)
    stds = np.std(DATA, axis=0)
    DATANORM = (data - means) / (stds + 1e-6)
    pca = PCA(n_components=4)
    out = pca.fit_transform(DATANORM)
    clustering = DBSCAN(eps=2, min_samples=4).fit(out)
    unique_labels = len(np.unique(clustering.labels_))
    print(f"Found {unique_labels} labels: {clustering.labels_}")




if __name__ == "__main__":
     main('/app/Data/WR/WR5_Run4.hdf5', '/app/Data/WR/Annotations/260218_annotations_a.pkl')