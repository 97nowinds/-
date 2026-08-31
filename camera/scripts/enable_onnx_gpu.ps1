param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $venvRoot = Join-Path $projectRoot ".venv"
    if (-not (Test-Path -LiteralPath $venvRoot)) {
        $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
    }
    $Python = Join-Path $venvRoot "Scripts\python.exe"
}
if (-not $venvRoot) {
    $venvRoot = Split-Path -Parent (Split-Path -Parent $Python)
}
& $Python -m pip uninstall -y onnxruntime onnxruntime-gpu
$cuda11Index = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"
& $Python -m pip install --disable-pip-version-check --no-cache-dir --no-deps `
    nvidia-cuda-runtime-cu11==11.8.89 `
    nvidia-cuda-nvrtc-cu11==11.8.89 `
    nvidia-cublas-cu11==11.11.3.6 `
    nvidia-cufft-cu11==10.9.0.58 `
    nvidia-cudnn-cu11==8.9.5.29
& $Python -m pip install --disable-pip-version-check --no-cache-dir --no-deps `
    onnxruntime-gpu==1.19.2 --index-url $cuda11Index
$nvidiaBinDirs = Get-ChildItem -LiteralPath (Join-Path $venvRoot "Lib\site-packages\nvidia") -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "bin" } |
    Select-Object -ExpandProperty FullName
foreach ($nvidiaBinDir in $nvidiaBinDirs) {
    $env:Path = "$nvidiaBinDir;$env:Path"
}
& $Python -c "import onnxruntime as ort; providers=ort.get_available_providers(); print(providers); assert 'CUDAExecutionProvider' in providers, 'CUDAExecutionProvider is unavailable'"
