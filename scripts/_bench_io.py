import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.ssl_cross_channel import CrossChannelMaskRasterDataset

idx = "data/processed/ssl_unlabeled_indices/unlabeled_patch_index_ps32_n20000.csv"
rd = "data/processed/rasters_cleaned"


def main():
    for nw in (4, 8):
        ds = CrossChannelMaskRasterDataset(idx, rd, normalize=False)
        N = 2000
        loader = DataLoader(Subset(ds, list(range(N))), batch_size=64, shuffle=False,
                            num_workers=nw, persistent_workers=False)
        t = time.time(); n = 0
        for X in loader:
            n += X.shape[0]
        dt = time.time() - t
        ds.close()
        print(f"num_workers={nw}: {n/dt:.0f} patches/s => {20000/(n/dt)/60:.1f} min/epoch => 50ep {50*20000/(n/dt)/3600:.1f}h", flush=True)

    ds = CrossChannelMaskRasterDataset(idx, rd, normalize=False)
    t = time.time()
    arr = np.stack([ds[i].numpy() for i in range(len(ds))]).astype("float32")
    ds.close()
    build = time.time() - t
    print(f"\nbuild full RAM cache: {len(arr)} patches in {build/60:.1f} min, {arr.nbytes/1e9:.2f} GB", flush=True)

    loader = DataLoader(TensorDataset(torch.from_numpy(arr)), batch_size=64, shuffle=True, num_workers=0)
    t = time.time(); n = 0
    for (X,) in loader:
        n += X.shape[0]
    dt = time.time() - t
    print(f"cached epoch (data only): {n/dt:.0f} patches/s => 50 ep ~ {50*dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
