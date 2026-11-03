#requires -Version 5.1
<#
.SYNOPSIS
  Split an SDXL checkpoint into Diffusers components and convert its UNet to GGUF.

.DESCRIPTION
  Output layout:
    diffusion_model.gguf  - SDXL UNet, converted by stable-diffusion.cpp
    clip_l.safetensors    - CLIP-L text encoder
    clip_g.safetensors    - CLIP-G text encoder
    vae.safetensors       - SDXL VAE
    tokenizer/            - both SDXL tokenizer folders

  GGUF is used for the diffusion/UNet component. stable-diffusion.cpp loads
  SDXL CLIP-L, CLIP-G and VAE as separate safetensors files; converting those
  three to GGUF is not part of this workflow.

.PARAMETER ModelPath
  A single-file SDXL .safetensors/.ckpt checkpoint, or a Diffusers SDXL folder.

.PARAMETER SdCli
  Path to the stable-diffusion.cpp sd-cli.exe built with SDXL support.

.PARAMETER OutputDir
  Destination directory.

.PARAMETER QuantType
  GGUF quantization type passed to sd-cli (default q8_0).

.PARAMETER Config
  Diffusers config/model id used by from_single_file. Defaults to the official
  SDXL Base 1.0 repository. For fine-tunes with a custom config, pass a
  compatible SDXL Diffusers config/repository.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$SdCli,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateSet("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")]
    [string]$QuantType = "q8_0",

    [string]$Config = "stabilityai/stable-diffusion-xl-base-1.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedCli = (Resolve-Path -LiteralPath $SdCli).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDir).Path
$workDir = Join-Path $resolvedOutput "_sdxl_components"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

# Verify the converter early, before doing the expensive checkpoint extraction.
if (-not (Test-Path -LiteralPath $resolvedCli -PathType Leaf)) {
    throw "sd-cli executable not found: $resolvedCli"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python 3 was not found on PATH. Install Python 3.10+." }

$extractor = Join-Path $workDir "extract_sdxl_components.py"
$pythonSource = @'
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--config", required=True)
p.add_argument("--is-file", action="store_true")
a = p.parse_args()

model = Path(a.model)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

dtype = torch.float16
print("Loading SDXL pipeline (this may require substantial system RAM)...", flush=True)
if a.is_file:
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model),
        config=a.config,
        torch_dtype=dtype,
        use_safetensors=model.suffix.lower() == ".safetensors",
    )
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )

required = ("unet", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2")
for name in required:
    if getattr(pipe, name, None) is None:
        raise RuntimeError(f"Input is not a complete SDXL pipeline: missing {name}")

# Keep components separate. A large shard limit makes a single safetensors
# file for each encoder/UNet, which sd-cli can consume directly.
pipe.unet.save_pretrained(out / "unet", safe_serialization=True, max_shard_size="50GB")
pipe.text_encoder.save_pretrained(out / "clip_l", safe_serialization=True, max_shard_size="50GB")
pipe.text_encoder_2.save_pretrained(out / "clip_g", safe_serialization=True, max_shard_size="50GB")
pipe.vae.save_pretrained(out / "vae", safe_serialization=True, max_shard_size="50GB")
pipe.tokenizer.save_pretrained(out / "tokenizer" / "clip_l")
pipe.tokenizer_2.save_pretrained(out / "tokenizer" / "clip_g")

print("Component extraction complete.", flush=True)
'@

[System.IO.File]::WriteAllText($extractor, $pythonSource, [System.Text.UTF8Encoding]::new($false))

$isFile = -not (Test-Path -LiteralPath $resolvedModel -PathType Container)
if ($isFile -and ([IO.Path]::GetExtension($resolvedModel).ToLowerInvariant() -notin @(".safetensors", ".ckpt", ".pt"))) {
    throw "Expected an SDXL .safetensors/.ckpt file or a Diffusers model directory."
}

Write-Host "Step 1/2: Extracting SDXL components..."
$pythonArgs = @(
    $extractor,
    "--model", $resolvedModel,
    "--output", $workDir,
    "--config", $Config
)
if ($isFile) { $pythonArgs += "--is-file" }
& $python.Source @pythonArgs
if ($LASTEXITCODE -ne 0) { throw "Diffusers component extraction failed (exit code $LASTEXITCODE)." }

$unetFile = Join-Path $workDir "unet/diffusion_pytorch_model.safetensors"
$clipLFile = Join-Path $workDir "clip_l/model.safetensors"
$clipGFile = Join-Path $workDir "clip_g/model.safetensors"
$vaeFile = Join-Path $workDir "vae/diffusion_pytorch_model.safetensors"
foreach ($file in @($unetFile, $clipLFile, $clipGFile, $vaeFile)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Expected component file was not produced: $file. The input may use an unsupported/sharded layout."
    }
}

Write-Host "Step 2/2: Converting SDXL UNet to GGUF ($QuantType)..."
$ggufFile = Join-Path $resolvedOutput "diffusion_model.gguf"
& $resolvedCli -M convert -m $unetFile -o $ggufFile -v --type $QuantType
if ($LASTEXITCODE -ne 0) { throw "sd-cli GGUF conversion failed (exit code $LASTEXITCODE)." }

Copy-Item -LiteralPath $clipLFile -Destination (Join-Path $resolvedOutput "clip_l.safetensors") -Force
Copy-Item -LiteralPath $clipGFile -Destination (Join-Path $resolvedOutput "clip_g.safetensors") -Force
Copy-Item -LiteralPath $vaeFile -Destination (Join-Path $resolvedOutput "vae.safetensors") -Force
Copy-Item -LiteralPath (Join-Path $workDir "tokenizer") -Destination (Join-Path $resolvedOutput "tokenizer") -Recurse -Force

Write-Host ""
Write-Host "Done. SDXL components are in: $resolvedOutput"
Write-Host "  diffusion_model.gguf  ($QuantType)"
Write-Host "  clip_l.safetensors"
Write-Host "  clip_g.safetensors"
Write-Host "  vae.safetensors"
Write-Host "  tokenizer/clip_l and tokenizer/clip_g"
Write-Warning "The CLIP and VAE components intentionally remain safetensors; stable-diffusion.cpp loads them separately. This script prepares model files only and does not yet add SDXL GGUF runtime support to Local Dream."
