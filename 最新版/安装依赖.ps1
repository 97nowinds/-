param(
    [string]$PythonExe = "",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $LocalPython = Join-Path $PackageRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $LocalPython)) {
        Write-Host "未指定 Python，正在包内创建 Python 3.11 虚拟环境 .venv"
        & py -3.11 -m venv (Join-Path $PackageRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "创建 Python 3.11 虚拟环境失败" }
    }
    $PythonExe = $LocalPython
}

$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
Write-Host "依赖将安装到：$PythonExe"
& $PythonExe -c "import sys; assert sys.version_info[:2] == (3, 11), '必须使用 Python 3.11'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python版本检查失败" }

& $PythonExe -m pip install --upgrade "pip==26.2.1" "setuptools==78.1.0"
if ($LASTEXITCODE -ne 0) { throw "pip升级失败" }

if (-not $SkipTorch) {
    & $PythonExe -m pip install --index-url "https://download.pytorch.org/whl/cu128" "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
    if ($LASTEXITCODE -ne 0) { throw "GPU版PyTorch安装失败" }
} else {
    Write-Host "已跳过PyTorch安装；稍后必须运行检查环境.py确认现有PyTorch可用CUDA。"
}

& $PythonExe -m pip install -r (Join-Path $PackageRoot "requirements-windows-lock.txt")
if ($LASTEXITCODE -ne 0) { throw "其余依赖安装失败" }

& $PythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip依赖一致性检查失败" }

Write-Host "安装完成。下一步执行："
Write-Host "& '$PythonExe' '$PackageRoot\检查环境.py'"
Write-Host "& '$PythonExe' '$PackageRoot\检查环境.py' --load-models"

