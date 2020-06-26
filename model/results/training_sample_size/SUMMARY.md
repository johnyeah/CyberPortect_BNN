Best overall results are achieved with the following hyperparameters:
```python
batch_size = 32
network_size = [20, 20]
sampling_size >= length(all_data) # preferably >= 1e5


```
## Experiment
### Batch Size
```python
batch_size in [16, 32, 64, 128, 256]
results = [79, 80, 79, 67, 60]
```
The best batch size is `32`. The results (for 50 iterations, [20 20] layers) are respectively
### Training data sampling
The more the sampling, the better the result. The results (for 50 iterations, [20 20] layers) are respectively

`sampling in [10, 100, 1000, 10000]`
### Number of layers

The more layers there are, the longer the training time (and more noise is introduced)

### Number of n_hidden_neurons
The more neurons there are, the longer the training time (and more noise is introduced)
 `network_size in [[64], [128], [256] * 1, [256, 128], [128, 32]]`

