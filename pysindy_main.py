import pysindy as ps
from pysindy import FourierLibrary, PolynomialLibrary,ConcatLibrary, CustomLibrary
from pysindy.optimizers import STLSQ, SR3
import numpy as np
from scipy.signal import hilbert
from datasets import WRsmallepoch
from torch.utils.data import DataLoader
import torch.utils.data as data
import dill
from tqdm import tqdm
from PyEMD import EMD
from matplotlib import pyplot as plt


def main(data_file, annotation_file, model_save_path, sample_rate=5000):
    
    dataset = WRsmallepoch(
    data_file = data_file, 
    annotation_file = annotation_file,
    epoch_size=5,
    epoch_id_restriction=2, 
    sample_rate= sample_rate,
    single_channel_flag=False,
    EMD_flag=False,
    psd_flag=False,
    BP_flag=True
    )

    train_set_size = int(len(dataset) * 0.8)

    train_indices = list(range(train_set_size))
    test_indices = list(range(train_set_size, len(dataset)))

    train_loader = DataLoader(data.Subset(dataset,train_indices),batch_size=train_set_size)
    test_loader = DataLoader(data.Subset(dataset,test_indices),batch_size=200)

    def extract_real_component(signal):
        return np.abs(hilbert(signal))
    def extract_imaginary_component(signal):
        return(np.angle(hilbert(signal)))
    def sigmoid_activation250k(signal):
        return 1/(1 + 250000**(-signal))
    def sigmoid_activation25k(signal):
        return 1/(1 + 25000**(-signal))
    def linear_activation(signal):
        return 2.8*signal + 0.5
    hilbert_lib = CustomLibrary(library_functions=[extract_real_component,extract_imaginary_component])
    poly_lib = PolynomialLibrary(degree=2)
    fourier_lib = FourierLibrary(n_frequencies=1)
    full_lib = ConcatLibrary([poly_lib, hilbert_lib,fourier_lib])


    optimizer = STLSQ(alpha=1, threshold= 4)
    #optimizer = SR3()

    model = ps.SINDy(feature_library=full_lib,optimizer=optimizer,feature_names=['ch1_data_low', 'ch2_data_low','ch1_data_mid', 'ch2_data_mid','ch1_data_high', 'ch2_data_high'])

    for batch in tqdm(train_loader):
        epochs, _= batch
        epochs = epochs.cpu().detach().numpy()

        epochs = (epochs - np.mean(epochs,axis=1, keepdims=True))/(np.std(epochs, axis=1, keepdims=True)+1e-06)

        list_of_epochs = [epochs[i] for i in range(epochs.shape[0])]
        model.fit(list_of_epochs,t=1/100,multiple_trajectories=True)
    print('Final Equation Fit')
    model.print()


    for batch in tqdm(test_loader):
        epochs, _= batch
        
        epochs = epochs.cpu().detach().numpy()
        epochs = (epochs - np.mean(epochs,axis=1, keepdims=True))/(np.std(epochs, axis=1, keepdims=True)+1e-06)

        list_of_epochs = [epochs[i] for i in range(epochs.shape[0])]
        test_score = model.score(list_of_epochs,t=1/100,multiple_trajectories=True)
        print(f'Score on Test Set:{test_score}')

    t = np.arange(0.01,5,0.01)
    x_0 = epochs[0,0,:]

    out = model.simulate(x_0,t,integrator_kws={"method": "RK45", "rtol":1e-03, "atol":1e-03})

    plt.plot(t,out)
    plt.savefig('/app/Data/tmp/sindy_temp.png')


    mean_transform = np.mean(epochs,axis=1, keepdims=True)
    std_transform = np.std(epochs, axis=1, keepdims=True)+1e-06

    # Save variables to a dictionary
    data_to_save = {
        'model': model,
        'out': out,
        'epoch': epochs,
        'mean_transform': mean_transform,
        'std_transform': std_transform

    }

    with open(model_save_path,'wb') as f:
        dill.dump(data_to_save,f)

        

if __name__ == "__main__":
    main('/app/Data/WR/WR5_Run4.hdf5', 
         '/app/Data/WR/Annotations/260223_PYSINDy_annotations.pkl',
         '/app/Data/WR/pySINDy/260310_model.pkl'
         )