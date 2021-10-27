import numpy as np
from uniform_instance_gen import uni_instance_gen


j = 3 #20
m = 2 # 10
l = 0
h = 99
batch_size = 3 #100
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(data.shape)
np.save('generatedData{}_{}_Seed{}.npy'.format(j, m, seed), data)