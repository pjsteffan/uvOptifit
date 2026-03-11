import pysindy as ps
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import dill

with open('/app/Data/WR/pySINDy/260310_model.pkl','rb') as f:
    model = dill.load(f)


t = np.arange(0.01,4,0.01)
x_0 = np.array([0.000025,0.000025])


out = model.simulate(x_0,t)

plt.plot(t,out)
plt.savefig('/app/data/tmp/sindy_temp.png')