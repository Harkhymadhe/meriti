# Performance Analysis and Optimization

---

| Index | Configuration                                                           | Performance in seconds |
| ----- | ----------------------------------------------------------------------- | ---------------------- |
| 1     | Baseline                                                                | 420                    |
| 2     | Baseline - Wandb                                                        | 383 - 390              |
| 3     | 2 + Rework weight instantiation in DiceLoss `__init__` impl             | 382 - 385              |
| 4     | 3 + don't clear model residual + Model priming                          | 830                    |
| 5     | 2 + convert dataset to SQUASHFS format                                  | 117                    |
| 6     | 2 + eliminate augmentation                                              | 90                     |

- Priming the model increased GPU utilization (99.8% utilization), but also drastically increased the training time! Not good.
- Replacing the `__getitem__` method with this: ```python
def __getitem__(self, idx: int):
    return torch.randn(3, 224, 224), torch.randint(0, 10, (1, 224, 224))
``` INCREDIBLY sped up training by a factor of 25x! This suggests that optimizing the `__getitem__` method is likely to drastically improve throughput and GPU utilization.
- Converting the dataset to SQUASHFS format using `data.sh` showed my line of thinking was correct!
  - Now, from `385 seconds`, we train an epoch in about `132 seconds`! However, we are still not completely utilizing the GPU (at 96.9% utilization)!
  - Analysing the Nsys results, it appears that some batches are quickly processed on the CPU, as opposed to others.
  - Batch retreival time ranges from `~200 ms` to `~3.6 s`.
  - Shorter retrieval times for batches may imply simpler image processing or cache hits.
