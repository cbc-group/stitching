from mpi4py import MPI
import h5py
import numpy as np

n = 100000000
channels = 32
num_processes = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank

np.random.seed(rank)

with h5py.File("mpi_test.h5", "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
    dset = f.create_dataset("test", (channels, n), dtype=np.float32)
    for i in range(channels):
        if i % num_processes == rank:
            print(f"rank={rank}, i={i}")
            data = np.random.uniform(size=n)
            dset[i] = data
