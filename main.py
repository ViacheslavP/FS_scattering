import numpy as np
from novelfss import atomic_state, new_cloud, new_chain, mean_absorbtion
from matplotlib import pyplot as plt

nat = 5
nreps = 500

cloud = new_cloud(noa=nat)
cloud.shuffle()


chain = new_chain(noa=nat,  d=2*np.pi*7)
chain.shuffle()

freq = np.linspace(-5, 5, 200)

cloud_abs = mean_absorbtion(cloud, freq, nreps)
chain_abs = mean_absorbtion(chain, freq, nreps)

plt.plot(freq, cloud_abs, 'r-')
plt.plot(freq, chain_abs, 'b-')
plt.show()
