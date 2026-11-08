# ΠGDM vs. DPS for Diffusion-Based Image Restoration

Course project for *Generative Models for Images* (MVA Master's, Télécom Paris), by Gloire Linvani and Dorval Mobendza.

Diffusion Posterior Sampling (DPS) and Pseudoinverse-Guided Diffusion Models (ΠGDM) restore images with a single pretrained diffusion model, by steering its sampling toward images consistent with the degraded measurement. The report derives both methods from how each approximates the intractable likelihood of the measurement given the current noisy image. The notebook compares them with a pretrained FFHQ 256×256 diffusion model on inpainting, Gaussian deblurring with and without measurement noise, and a composed degradation (blur, random masking, and 4× downsampling).

> **Version note.** This is the version submitted for grading in March 2026, tagged [`submitted`](https://github.com/GloireLINVANI/pigdm-vs-dps/tree/submitted). A revised version, with every experiment re-run end to end, will follow.

## Contents

| File | Description |
|---|---|
| `report.pdf` | Report: derivations of both methods and experimental results |
| `PiGDM_DPS_experiments.ipynb` | Experiments: hyperparameter tuning on validation images, final evaluation on test images, tables and curves |
| `pigdm_dps_core.py` | Degradation operators, DPS and ΠGDM samplers, and metrics (PSNR, SSIM, LPIPS) |
| `requirements.txt` | Python dependencies |

## Running the experiments

The notebook was written for Google Colab with a GPU, and its comments are in French.

1. Open `PiGDM_DPS_experiments.ipynb` in Colab and upload `pigdm_dps_core.py` next to it.
2. Run the cells in order. The setup cells install the dependencies, clone the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling) for its `guided_diffusion` model code, and download the FFHQ 256×256 validation images and the pretrained checkpoint `ffhq_10m.pt` from copies shared for the course.

To run it locally instead, install the dependencies with `pip install -r requirements.txt` and start Jupyter from the repository root. The setup cell calls `git`, `wget`, and `unzip`.

## Checkpoint and validation images

If the course copies become unavailable, get both from their original sources and skip the matching download lines in the setup cell.

**Checkpoint.** The DPS authors released `ffhq_10m.pt`, the FFHQ 256×256 diffusion model used here, in a [Google Drive folder](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) linked from their repository's README. Save it as `ffhq_10m.pt` next to the notebook.

**Validation images.** The DPS authors trained this model on FFHQ images 01000 to 49999 and kept the first 1,000 images, `00000.png` to `00999.png`, for validation ([DPS issue #4](https://github.com/DPS2022/diffusion-posterior-sampling/issues/4)). To rebuild that set, download the folder `images1024x1024/00000` from the [official FFHQ release](https://github.com/NVlabs/ffhq-dataset), then resize its images to 256×256 into a folder `ffhq256-1k-validation`, keeping their file names:

```python
from pathlib import Path
from PIL import Image

src, dst = Path("00000"), Path("ffhq256-1k-validation")
dst.mkdir(exist_ok=True)
for path in sorted(src.glob("*.png")):
    Image.open(path).resize((256, 256), Image.Resampling.LANCZOS).save(dst / path.name)
```

The DPS authors did not document their resizing filter, so pixel values may differ slightly from theirs.

## References

- H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR 2023.
- J. Song, A. Vahdat, M. Mardani, and J. Kautz. *Pseudoinverse-Guided Diffusion Models for Inverse Problems.* ICLR 2023.

The `guided_diffusion` model code and the FFHQ checkpoint come from the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling).
