# PaDiM

Stage 3 of the anomaly-detection roadmap (see root `CLAUDE.md`). Unlike `autoencoder/` and
`variational_autoencoder/`, PaDiM is not trained by backpropagation — it's a statistical fit:
extract patch-level embeddings from a **frozen, pretrained** ImageNet CNN over the "good"
training images, then fit a multivariate Gaussian (mean + covariance) per patch position.
Anomaly scoring at test time is the Mahalanobis distance from each test patch's embedding to its
position's fitted Gaussian, which gives a per-pixel anomaly heatmap directly (no reconstructed
image involved).

## Files

- `dataset.py` — same `MVTecDataset` as the other stages, except images are loaded as RGB
  (`.convert('RGB')`) instead of grayscale, since a pretrained ImageNet backbone needs real
  3-channel color input to produce meaningful features.
- `model.py`
  - `FeatureExtractor`: frozen `torchvision.models.resnet18` (ImageNet weights), stops after
    `layer3` (skips `layer4`/avgpool/fc, which PaDiM doesn't need). Upsamples `layer2`/`layer3`
    to `layer1`'s resolution (nearest-neighbor) and concatenates channel-wise.
  - `PaDiM`: plain Python class (no learnable parameters, not an `nn.Module`) that owns the
    `FeatureExtractor`, a fixed random subset of feature channels (`d_reduced=100` of 448, per
    the paper's dimensionality-reduction step), and the fitted per-patch `mean`/`cov_inv`.
    - `fit(dataloader, device)` — single pass over training images, fully vectorized with
      `torch.einsum`/`torch.linalg.inv` (no per-patch Python loop).
    - `predict(x, device, image_size)` — returns `(image_scores, anomaly_maps)`.
    - `save(path)` / `load(path, device)` — persists the fitted stats + selected channel
      indices via `torch.save`/`torch.load`.
- `train.py` — no epoch loop: instantiate `PaDiM`, one `model.fit(...)` pass over
  `train/good`, save to `padim_{target_object}.pth` (no epoch suffix — nothing to checkpoint
  incrementally).
- `evaluate.py` — same overall structure as `variational_autoencoder/evaluate.py` (per-type
  anomaly loaders, logging to file, ROC curve + optimal threshold, confusion matrix,
  classification report, false-positive/false-negative filtering), but:
  - anomaly score = max of the per-pixel Mahalanobis distance map, instead of reconstruction
    error
  - `visualize_filtered_anomaly_maps` shows Original / Anomaly Heatmap / Overlay per image,
    instead of Original / Reconstructed / Error
- `requirements.txt` — same as the other stages, minus `pytorch-msssim` (no SSIM loss here).

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Backbone | resnet18, ImageNet-pretrained, frozen | Lighter than wide_resnet50_2; already explored once in this repo's history |
| Layers used | `layer1`, `layer2`, `layer3` | Standard PaDiM setup; `layer4`/fc dropped as unnecessary compute |
| Input size | 256×256, no center-crop | Consistent with how the AE/VAE stages already handle MVTec images |
| Feature grid | 64×64, 448 channels (64+128+256) | `layer1`'s resolution at 256×256 input |
| `d_reduced` | 100 | Paper's default for a resnet18 backbone |
| Covariance regularization | `+ 0.01 * I` | Paper's value, needed for invertibility |
| Image-level score | `max` of the anomaly map | Paper's choice; matches the single-scalar-per-image pattern the ROC/threshold code expects |
| Checkpoint naming | `padim_{target_object}.pth`, no epoch suffix | There are no epochs — one fit pass produces one artifact |

## Running it

Same environment assumptions as the other two stages: MVTec-style data expected under
`/content/mvtec/<target_object>/`. Additionally, the **first** run of `train.py` or
`evaluate.py` needs internet access once, to download the pretrained resnet18 ImageNet weights
(cached by `torchvision` after that).

```
cd padim
python train.py      # fits the Gaussians, saves saved_models/<object>/padim_<object>.pth
python evaluate.py    # scores the test set, writes plots + a log file to saved_models/<object>/
```
