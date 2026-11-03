# SDXL GGUF preparation (Windows)

`convert-sdxl-to-gguf.ps1` prepares a component-separated SDXL model package from either:

- a single-file SDXL checkpoint (`.safetensors` or `.ckpt`), or
- a local Diffusers SDXL pipeline directory.

It extracts the SDXL components with Hugging Face Diffusers, then converts the **UNet/diffusion model** to GGUF using `sd-cli` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

## Requirements

- Windows PowerShell 5.1 or PowerShell 7
- Python 3.10+ on `PATH`
- A Windows `sd-cli.exe` built with SDXL support
- Python packages: `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`

Install the Python dependencies in the environment that will run the script. Use a PyTorch build appropriate for your PC.

## Example

```powershell
Set-ExecutionPolicy -Scope Process Bypass

.\tools\convert-sdxl-to-gguf.ps1 `
  -ModelPath "D:\Models\sd_xl_base_1.0.safetensors" `
  -SdCli "D:\stable-diffusion.cpp\build\bin\Release\sd-cli.exe" `
  -OutputDir "D:\Models\sdxl-localdream" `
  -QuantType q8_0
```

For a local Diffusers directory, pass the directory as `-ModelPath`. The directory must contain a complete SDXL pipeline and is loaded offline.

For a fine-tuned single-file checkpoint whose architecture/config differs from SDXL Base 1.0, pass a compatible Diffusers config/repository with `-Config`. The default is `stabilityai/stable-diffusion-xl-base-1.0`.

## Output

| File | Component | Format |
|---|---|---|
| `diffusion_model.gguf` | SDXL UNet | GGUF, selected quantization |
| `clip_l.safetensors` | CLIP-L text encoder | Safetensors |
| `clip_g.safetensors` | CLIP-G text encoder | Safetensors |
| `vae.safetensors` | SDXL VAE | Safetensors |
| `tokenizer/clip_l/`, `tokenizer/clip_g/` | Tokenizers | Diffusers tokenizer files |

Only the UNet is converted to GGUF here. The SDXL path in stable-diffusion.cpp consumes CLIP-L, CLIP-G and VAE as separate component files; this script deliberately preserves those in safetensors rather than pretending every component has a supported GGUF conversion path.

The script leaves intermediate extracted files under `_sdxl_components` for diagnosis/re-running. The final package layout is a preparation artifact: Local Dream still needs its SDXL GGUF runtime/pipeline integration before these files can be selected and run in the app.

## Notes

- Checkpoint extraction can require substantial PC RAM and disk space. Keep enough free space for the original checkpoint, extracted components and final GGUF.
- `q8_0` is the default. Lower-bit quantization reduces model size but may affect image quality; choose the type according to the sd-cli build and target device.
- The converter must support loading a standalone Diffusers-format SDXL UNet. If it reports that the SD version cannot be detected, update/build stable-diffusion.cpp with standalone SDXL UNet detection before running this script.
