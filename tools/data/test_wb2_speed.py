"""Quick WB2 GCS read speed test."""
import time, gcsfs, zarr
import numpy as np

fs = gcsfs.GCSFileSystem(token='anon')
arr = zarr.open(fs.get_mapper('weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr/temperature'), mode='r')
print(f'shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}')

# Test 1: single hour, single level
t0 = time.time()
data = arr[534720, 36, :, :]
dt = time.time() - t0
mb = data.nbytes / 1024**2
print(f'1h × 1lvl: {dt:.2f}s for {mb:.2f}MB = {mb/dt:.2f}MB/s')

# Test 2: 1 hour, 4 levels (oindex)
t0 = time.time()
data = arr.oindex[534720:534721, [36, 33, 30, 25], :, :]
dt = time.time() - t0
mb = data.nbytes / 1024**2
print(f'1h × 4lvl (oindex): {dt:.2f}s for {mb:.2f}MB = {mb/dt:.2f}MB/s')

# Test 3: 24 hours, 1 level
t0 = time.time()
data = arr[534720:534744, 36, :, :]
dt = time.time() - t0
mb = data.nbytes / 1024**2
print(f'24h × 1lvl: {dt:.2f}s for {mb:.2f}MB = {mb/dt:.2f}MB/s')

# Test 4: 24 hours, 4 levels via oindex
t0 = time.time()
data = arr.oindex[534720:534744, [36, 33, 30, 25], :, :]
dt = time.time() - t0
mb = data.nbytes / 1024**2
print(f'24h × 4lvl (oindex): {dt:.2f}s for {mb:.2f}MB = {mb/dt:.2f}MB/s')
