import torch
import os
from os import path as osp
import numpy as np
import ml4reco_modules
from ml4reco_modules.extensions.binned_select_knn import _binned_select_knn 



torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'binned_select_knn_grad_cpu.so'))
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'binned_select_knn_grad_cuda.so'))

op = torch.ops.binned_select_knn_cuda.binned_select_knn_cuda #only test cuda version here

#copy from boiler plate around kernel
def prepare_inputs(K,
                coords: torch.Tensor,
                row_splits: torch.Tensor,
                direction=None, 
                n_bins=None, 
                max_bin_dims: int = 3, 
                torch_compatible_indices=False):


    # Estimate a good number of bins for homogeneous distributions
    elems_per_rs = torch.max(row_splits) / row_splits.shape[0]
    elems_per_rs = elems_per_rs.to(dtype=torch.int32) + 1

    # Limit max_bin_dims to the number of coordinate dimensions
    max_bin_dims = min(max_bin_dims, coords.shape[1])

    # Calculate n_bins if not provided
    if n_bins is None:
        n_bins = torch.pow(elems_per_rs.float() / (K / 32), 1. / float(max_bin_dims))
        n_bins = n_bins.to(dtype=torch.int32)
        n_bins = torch.where(n_bins < 5, torch.tensor(5, dtype=torch.int32), n_bins)
        n_bins = torch.where(n_bins > 30, torch.tensor(30, dtype=torch.int32), n_bins)

    # Handle binning for the coordinates
    bin_coords = coords
    if bin_coords.shape[-1] > max_bin_dims:
        bin_coords = bin_coords[:, :max_bin_dims]  # Truncate the extra dimensions

    # Call BinByCoordinates to assign bins
    dbinning, binning, nb, bin_width, nper = ml4reco_modules.bin_by_coordinates(bin_coords, row_splits, n_bins=n_bins)

    # Sort the points by bin assignment
    sorting = torch.argsort(binning, dim=0)
    #cast sorting to int32
    sorting = sorting.to(dtype=torch.int32)

    # Gather sorted coordinates and bin information
    scoords = coords[sorting]
    sbinning = binning[sorting]
    sdbinning = dbinning[sorting]

    if direction is not None:
        direction = direction[sorting]

    # Create bin boundaries (cumulative sum of number of points per bin)
    bin_boundaries = torch.cat([torch.zeros(1, dtype=torch.int32).to(coords.device), nper], dim=0)
    bin_boundaries = torch.cumsum(bin_boundaries, dim=0, dtype=torch.int32)

    # Ensure the bin boundaries are valid
    assert torch.max(bin_boundaries) == torch.max(row_splits), "Bin boundaries do not match row splits."

    direction = torch.empty(0, device=coords.device, dtype=binning.dtype)

    return scoords, sbinning, sdbinning, bin_boundaries, nb, bin_width, direction, torch_compatible_indices


def create_data(K, N, N_coord_dim):
    torch.manual_seed(45) 

    coordinates = torch.rand((N, N_coord_dim), dtype=torch.float32, device='cpu').to('cuda')
    row_splits = torch.tensor([0, N], dtype=torch.int32, device='cuda')

    return prepare_inputs(K, coordinates, row_splits)


def run_perf_test(Ns = [10000, 100000],
                  dims = [3, 8, 10],
                  K= [32, 64, 128, 256]):
    
    def run_test(N, N_coord_dim, K):
        data = create_data(K, N, N_coord_dim)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = op(*data, False, K)
        torch.cuda.synchronize()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1)
    
    results = {}
    for n in Ns:
        for d in dims:
            for k in K:
                print(f"Running test for N={n}, dim={d}, K={k}")
                run_test(n, d, k) #warmup
                t = 0
                for _ in range(5):
                    t += run_test(n, d, k)
                results[(n, d, k)] = t / 5
                print('took', results[(n, d, k)], 'ms')

    #save the results as pickle
    import pickle
    with open('binned_knn_perf_results.pkl', 'wb') as f:
        pickle.dump(results, f)


def make_plot():
    #plot all results
    import pickle
    import matplotlib.pyplot as plt
    with open('binned_knn_perf_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # make one plot per N, and different colors per dim. xaxes in all plots should be K
    
    
    for n in [10000, 100000]:
        fig, ax = plt.subplots()
        for d in [3, 8, 10]:
            x = [k for k in [32, 64, 128, 256]]
            y = [results[(n, d, k)] for k in x]
            ax.plot(x, y, label=f'N={n}, dim={d}')
        
        ax.set_title(f'Performance of binned_knn for N={n}')
        ax.set_xlabel('K')
        ax.set_ylabel('time (ms)')
        ax.legend()

        #set y range to 0 to 7000
        ax.set_ylim(0, 7000)

        plt.savefig(f'binned_knn_perf_N{n}.png')

    

#run_perf_test()
make_plot()