param(
    [string]$BackendUrl = "http://127.0.0.1:15000",
    [int]$Port = 5173,
    [switch]$ConfigureMemberIntegration,
    [switch]$GuiTokenPrompt,
    [switch]$DisableLivePublisher,
    [int]$LiveMaxFrames = 0,
    [string]$MemberServiceUrl = "http://192.168.31.189:8765"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
}
$python = Join-Path $venvRoot "Scripts\python.exe"
$server = Join-Path $projectRoot "frontend_server.py"

function Read-MemberTokenFromDialog {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "成员接收端 Token"
    $form.Size = New-Object System.Drawing.Size(520, 175)
    $form.StartPosition = "CenterScreen"
    $form.TopMost = $true
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false

    $label = New-Object System.Windows.Forms.Label
    $label.Text = "请输入成员接收端临时 Token（内容会隐藏且不会保存到文件）："
    $label.Location = New-Object System.Drawing.Point(18, 18)
    $label.AutoSize = $true
    $form.Controls.Add($label)

    $textBox = New-Object System.Windows.Forms.TextBox
    $textBox.Location = New-Object System.Drawing.Point(20, 48)
    $textBox.Size = New-Object System.Drawing.Size(465, 25)
    $textBox.UseSystemPasswordChar = $true
    $form.Controls.Add($textBox)

    $okButton = New-Object System.Windows.Forms.Button
    $okButton.Text = "确定"
    $okButton.Location = New-Object System.Drawing.Point(325, 88)
    $okButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.AcceptButton = $okButton
    $form.Controls.Add($okButton)

    $cancelButton = New-Object System.Windows.Forms.Button
    $cancelButton.Text = "取消"
    $cancelButton.Location = New-Object System.Drawing.Point(410, 88)
    $cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.CancelButton = $cancelButton
    $form.Controls.Add($cancelButton)

    $form.Add_Shown({ $textBox.Select() })
    $result = $form.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        $form.Dispose()
        throw "已取消成员接收端 Token 输入"
    }
    $token = $textBox.Text
    $textBox.Clear()
    $form.Dispose()
    return $token
}

if ($ConfigureMemberIntegration) {
    $env:LAB_MEMBER_SERVICE_URL = $MemberServiceUrl
    if ([string]::IsNullOrWhiteSpace($env:LAB_MEMBER_SERVICE_TOKEN)) {
        if ($GuiTokenPrompt) {
            $env:LAB_MEMBER_SERVICE_TOKEN = Read-MemberTokenFromDialog
        }
        else {
            $secureToken = Read-Host "请输入成员接收端临时 Token（输入不会显示）" -AsSecureString
            $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
            try {
                $env:LAB_MEMBER_SERVICE_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
            }
            finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
                $secureToken.Dispose()
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($env:LAB_MEMBER_SERVICE_TOKEN)) {
        throw "成员接收端 Token 不能为空"
    }
    $env:LAB_MEMBER_SERVICE_TIMEOUT_SECONDS = "5"
    if ($DisableLivePublisher) {
        $env:LAB_MEMBER_LIVE_PUBLISH_ENABLED = "0"
    }
    if ($LiveMaxFrames -gt 0) {
        $env:LAB_MEMBER_LIVE_PUBLISH_ENABLED = "1"
        $env:LAB_MEMBER_LIVE_MAX_FRAMES = [string]$LiveMaxFrames
    }
}

Set-Location $projectRoot
Write-Host "Standalone laboratory frontend" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray
Write-Host "Backend API: $BackendUrl" -ForegroundColor Yellow
Write-Host "Frontend: http://127.0.0.1:$Port/?api=$BackendUrl" -ForegroundColor Green
if ($ConfigureMemberIntegration) {
    Write-Host "Member integration: $MemberServiceUrl (Bearer Token 已隐藏)" -ForegroundColor Green
}
& $python $server --host 0.0.0.0 --port $Port --api $BackendUrl
