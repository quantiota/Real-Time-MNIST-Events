# N-MNIST Event Camera Dataset

Neuromorphic MNIST recorded by an ATIS event camera saccading over MNIST digits displayed on a monitor.
Each sample is a stream of `(x, y, timestamp, polarity)` events — not frames.

- **Source**: Orchard et al., 2015 — *Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades*
- **Sensor**: ATIS event camera (34x34 pixels)
- **Events per sample**: ~4000-6000
- **Duration per sample**: ~300 ms (3 saccades)
- **Polarity**: 1 = brightness increase, 0 = brightness decrease

## Download

```bash
pip install tonic
```

```python
import tonic

# Train set (60,000 samples, ~1.3 GB extracted)
dataset = tonic.datasets.NMNIST(save_to='/home/coder/project/Real-Time-MNIST-Events', train=True)

# Test set (10,000 samples)
dataset = tonic.datasets.NMNIST(save_to='/home/coder/project/Real-Time-MNIST-Events', train=False)
```

## Structure

```
NMNIST/
  Train/
    0/    # 5923 samples
    1/    # 6742 samples
    ...
    9/
  train.zip   # 966 MB (can be deleted after extraction)
```

## Load a sample

```python
import tonic

dataset = tonic.datasets.NMNIST(save_to='/home/coder/project/Real-Time-MNIST-Events', train=True)
events, label = dataset[0]

# events dtype: [('x', int64), ('y', int64), ('t', int64), ('p', int64)]
# label: 0-9
```

## Reference

Orchard, G., Jayawant, A., Cohen, G.K. and Thakor, N., 2015.
*Converting static image datasets to spiking neuromorphic datasets using saccades.*
Frontiers in neuroscience, 9, p.437.
