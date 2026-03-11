import dill 
from datasets import WRsmallepoch
import pysindy as ps
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt,welch, coherence
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.utils.data as data
import dill


def filter_data(data, lowcut=5, highcut=30, fs=100.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


with open('/app/Data/WR/pySINDy/260310_model.pkl','rb') as f:
    all = dill.load(f)

model = all['model']
out = all['out']
epoch = all['epoch']

mean_transform = all['mean_transform']
std_transform = all['std_transform']

dataset = WRsmallepoch(
data_file = '/app/Data/WR/WR5_Run4.hdf5', 
annotation_file = '/app/Data/WR/Annotations/260223_PYSINDy_annotations.pkl',
epoch_size=5,
epoch_id_restriction=2, 
sample_rate= 5000,
single_channel_flag=False,
EMD_flag=False,
psd_flag=False,
BP_flag=False
    )

train_set_size = int(len(dataset) * 0.8)

train_indices = list(range(train_set_size))
test_indices = list(range(train_set_size, len(dataset)))

train_loader = DataLoader(data.Subset(dataset,train_indices),batch_size=train_set_size)
test_loader = DataLoader(data.Subset(dataset,test_indices),batch_size=200)

data_batch = next(iter(test_loader))
comparison_epochs, _= data_batch


t = np.arange(0.01,5,0.01)

PXX_ch1 = []
PXX_exp = []
CXY = []
for i in tqdm(range(epoch.shape[0])):
        
    x_0 = epoch[i,0,:]

  
    out = model.simulate(x_0,t,integrator_kws={"method": "RK45", "rtol":1e-03, "atol":1e-03})


    out = out * std_transform[0,:,:] + mean_transform[0,:,:]

    channel1 = out[:,0]+out[:,2]+out[:,4]
    channel2 = out[:,1]+out[:,3]+out[:,5]

    example = comparison_epochs[i,:,0].cpu().detach().numpy()

    channel1 = filter_data(channel1, lowcut=5, highcut=30, fs=100.0, order=5)


    example = example[:499]


    # Calculate the power spectral density for both datasets
    f, Pxx_ch1 = welch(channel1, 100, nperseg=64)
    f, Pxx_exp = welch(example, 100, nperseg=64)
    PXX_ch1.append(Pxx_ch1)
    PXX_exp.append(Pxx_exp)


    # Calculate the coherence between channel1 and example
    f_cohere, Cxy = coherence(channel1, example, 100, nperseg=64)
    CXY.append(Cxy)

    # Calculate the mean of the power spectral densities across iterations
    PXX_ch1_mean = np.mean(PXX_ch1, axis=0)
    PXX_exp_mean = np.mean(PXX_exp, axis=0)

    # Calculate the standard deviation of the power spectral densities
    PXX_ch1_std = np.std(PXX_ch1, axis=0)
    PXX_exp_std = np.std(PXX_exp, axis=0)

    # Calculate the 95% confidence intervals
    confidence_interval_ch1 = 1.96 * PXX_ch1_std / np.sqrt(len(PXX_ch1))
    confidence_interval_exp = 1.96 * PXX_exp_std / np.sqrt(len(PXX_exp))

    # Plot the mean power spectral densities with confidence intervals
    plt.clf()
    plt.semilogy(f[4:-13], PXX_ch1_mean[4:-13]/np.max(PXX_ch1_mean[4:-13]), label='Channel 1')
    plt.fill_between(f[4:-13], 
                    (PXX_ch1_mean[4:-13] - confidence_interval_ch1[4:-13])/np.max(PXX_ch1_mean[4:-13]), 
                    (PXX_ch1_mean[4:-13] + confidence_interval_ch1[4:-13])/np.max(PXX_ch1_mean[4:-13]), 
                    alpha=0.2)

    plt.semilogy(f[4:-13], PXX_exp_mean[4:-13]/np.max(PXX_ch1_mean[4:-13]), label='Example')
    plt.fill_between(f[4:-13], 
                    (PXX_exp_mean[4:-13] - confidence_interval_exp[4:-13])/np.max(PXX_ch1_mean[4:-13]), 
                    (PXX_exp_mean[4:-13] + confidence_interval_exp[4:-13])/np.max(PXX_ch1_mean[4:-13]), 
                    alpha=0.2)

    plt.title(f'Average Power Spectral Density at iteration {i} with 95% Confidence Intervals')
    plt.xlabel('frequency [Hz]')
    plt.legend()
    plt.savefig('/app/Data/store/average_psd.png')
    


    # Calculate the mean coherence across iterations
    CXY_mean = np.mean(CXY, axis=0)

    # Calculate the standard deviation of the coherence
    CXY_std = np.std(CXY, axis=0)

    # Calculate the 95% confidence intervals
    confidence_interval_coh = 1.96 * CXY_std / np.sqrt(len(CXY))

    # Plot the mean coherence with confidence intervals
    plt.clf()
    plt.semilogy(f_cohere[4:-13], CXY_mean[4:-13], label='Coherence')
    plt.fill_between(f_cohere[4:-13], 
                    CXY_mean[4:-13] - confidence_interval_coh[4:-13], 
                    CXY_mean[4:-13] + confidence_interval_coh[4:-13], 
                    alpha=0.2)

    plt.title(f'Average Coherence at iteration {i} with 95% Confidence Intervals')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.legend()
    plt.savefig('/app/Data/store/average_coherence.png')

# Save variables to a dictionary
data_to_save = {
    'PXX_ch1': PXX_ch1,
    'PXX_exp': PXX_exp,
    'CXY': CXY,
}

with open('/app/Data/store/lists.pkl','wb') as f:
    dill.dump(data_to_save,f)



