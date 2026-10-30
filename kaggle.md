# Kaggle: MOSS-Image SFT LoRA on Two T4 GPUs

The repository includes [`notebooks/Moss-Image-SFT-Kaggle-Dual-GPU.ipynb`](notebooks/Moss-Image-SFT-Kaggle-Dual-GPU.ipynb), a Kaggle notebook for training MOSS-Image SFT LoRAs with the `Moss-Image SFT-Kaggle` architecture entry.

## Before starting

1. Open the notebook in Kaggle and set **Accelerator** to **GPU T4 x2**.
2. Enable **Internet** so the notebook can clone this repository and download model files.
3. Add a Kaggle Dataset containing your training images. Matching `.txt` caption files are optional; images and captions should have the same filename stem.
4. Run the notebook's **Setup** cell. It checks that two GPUs are visible, clones this fork's `main` branch, installs the toolkit requirements while keeping Kaggle's CUDA-enabled PyTorch, and initializes temporary model/cache directories.

## Configure and train

In **Settings & Dataset Preparation**, set:

- `DATASET_DIR` to the mounted dataset path, such as `/kaggle/input/my-image-dataset`.
- `LORA_NAME` to the desired run/checkpoint folder name.
- `TRIGGER_WORD` if prompts should activate a specific subject or style.
- `RANK`, `LR`, `TRAIN_STEPS`, and `SAVE_EVERY` for the training run.
- `RESOLUTIONS` to `[1024]` by default. Change it to `[512]` if either T4 runs out of VRAM.

The notebook selects `MODEL_ARCH = "moss_image_kaggle"` and `MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"`. It enables ConvRot8 quantization, cached latents and text embeddings, gradient checkpointing, and fp16 training for T4 GPUs.

Run the **Train / Resume LoRA** cell to write the configuration and launch:

```bash
accelerate launch --multi_gpu --num_processes 2 run.py /kaggle/working/configs/<LORA_NAME>.yaml
```

The notebook runs that command for you and reports VRAM for both GPUs, system RAM, and the latest log line. Training logs are written to `/kaggle/working/logs/`.

## How the two-GPU mode works

This is **data parallel**, not model parallel. Each GPU loads a complete copy of MOSS-Image SFT, processes a different batch, and averages LoRA gradients before the optimizer update. With the notebook defaults (`batch_size=1` on each worker), each optimizer step sees two samples total.

Each GPU must have enough VRAM for its own full model and training activations. The GPUs' memory is not combined, and this mode will not make a model that is too large for one T4 fit. The initial implementation supports exactly two CUDA GPUs, LoRA training, `train_unet: true`, `train_text_encoder: false`, and `model.layer_offloading: false`. Keep using the normal `MOSS-Image SFT` architecture for single-GPU runs.

## Checkpoints and resume

The toolkit writes checkpoints under `/kaggle/working/output/<LORA_NAME>`. The **Package LoRA** cell collects the `.safetensors` files into `/kaggle/working/lora_out.zip` for download.

To continue in a later Kaggle session, attach the previous output as a Kaggle input, set `RESUME_FROM` to the saved run folder (including `optimizer.pt`) or a checkpoint file, keep `LORA_NAME` and `RANK` the same, and raise `TRAIN_STEPS` to the desired cumulative step count. A folder resume can restore optimizer state when that file is present; a single LoRA checkpoint may resume without the previous optimizer state.

## Troubleshooting

- **Only one GPU is listed:** select **GPU T4 x2** in notebook settings, then restart the session and rerun Setup.
- **`moss_image_kaggle` is unknown:** make sure the notebook cloned the fork's current `main` branch; rerun Setup after the changes are pushed.
- **CUDA out of memory:** set `RESOLUTIONS = [512]`, reduce `RANK`, or lower the training batch size on both workers equally.
- **Early exit:** inspect `/kaggle/working/logs/<LORA_NAME>_train.log` for the distributed launch error. The implementation requires the NCCL backend, supplied by the standard Linux Kaggle GPU runtime.
- **To stop training:** interrupt the training cell. It signals the distributed process group and leaves completed checkpoints in the output folder.
