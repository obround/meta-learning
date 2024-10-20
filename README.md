# meta-learning


## Overview
I have been working on a side project over the past few days. Along the way, I implemented the following meta-learning algorithms in pytorch:
 * MAML ([paper](https://arxiv.org/abs/1703.03400))
 * MAML++ ([paper](https://arxiv.org/abs/1810.09502))
 * Reptile ([paper](https://arxiv.org/abs/1803.02999))
 * MetaSGD ([paper](https://arxiv.org/abs/1707.09835))

The MAML++ implementation only includes the multi-step loss optimization meant to improve gradient stability.

These algorithms are toy implementations. The code is decently documented if you want to take a look, but `meta_learning_algorithms.py` lacks abstractions across the algorithm implementations.

## Example: MetaSGD on the Sinusoid Dataset

![output](https://github.com/user-attachments/assets/027c036d-68df-4e1e-a692-f0f55ee1c102)

## Requirements
 * `torch==2.5.0`
 * `higher==0.2.1` for MAML and MAML++

## Example Usage
```python
from meta_learning_algorithms import MAML, Reptile, MetaSGD

...  # loading datasets, creating the models, etc.

maml = MAML(
    model=maml_model,
    loss=nn.MSELoss(),
    maml_plus_plus=False,
    inner_lr=1e-2,
    meta_lr=1e-1,
    device="cpu"
)
maml_plus_plus = MAML(
    model=maml_pp_model,
    loss=nn.MSELoss(),
    maml_plus_plus=True,
    inner_lr=1e-2,
    meta_lr=1e-1,
    device="cpu"
)
meta_sgd = MetaSGD(
    model=meta_sgd_model,
    loss=nn.MSELoss(),
    inner_lr=1e-3,
    meta_lr=1e-3,
    device="cpu"
)
reptile = Reptile(
    model=reptile_model,
    loss=nn.MSELoss(),
    inner_lr=1e-1,
    meta_lr=1e-1,
    clipping=4.0,
    device="cpu"
)
```
and then training would look like
```python
maml.train(train_data_loader, val_data_loader, epochs=15)
```

## Random Footnotes
 * The `MetaSGD` implementation uses the `functional_call` from `torch.func` instead of the `higher` monkey-patched functional module function. I did not realize that pytorch had such a function-- this makes it way easier to do meta-learning
 * The `get_per_step_loss_importance_vector` in MAML++ is taken from the How To Train Your MAML source code. Maybe I didn't read the paper close enough, but I could not find how they calculated $v_i$ in the paper
 * The code is not abstracted well because I originally wrote these in a set of messy jupyter notebooks before putting them together
