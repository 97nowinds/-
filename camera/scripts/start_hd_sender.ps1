$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$senderRoot = Split-Path -Parent $PSScriptRoot
$form = New-Object Windows.Forms.Form
$form.Text = '高清 cam_2 连续发送：联调地址与令牌'
$form.Size = New-Object Drawing.Size(620,260)
$form.StartPosition = 'CenterScreen'
$form.TopMost = $true
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$urlLabel = New-Object Windows.Forms.Label
$urlLabel.Text = '接收服务地址（如有变化请修改）：'
$urlLabel.Location = New-Object Drawing.Point(20,15)
$urlLabel.AutoSize = $true
$urlBox = New-Object Windows.Forms.TextBox
$urlBox.Location = New-Object Drawing.Point(20,40)
$urlBox.Size = New-Object Drawing.Size(560,25)
$urlBox.Text = 'http://100.66.129.119:8765'
$tokenLabel = New-Object Windows.Forms.Label
$tokenLabel.Text = '新联调令牌（隐藏输入，只传入进程内存）：'
$tokenLabel.Location = New-Object Drawing.Point(20,80)
$tokenLabel.AutoSize = $true
$tokenBox = New-Object Windows.Forms.TextBox
$tokenBox.Location = New-Object Drawing.Point(20,105)
$tokenBox.Size = New-Object Drawing.Size(560,25)
$tokenBox.UseSystemPasswordChar = $true
$ok = New-Object Windows.Forms.Button
$ok.Text = '开始发送'
$ok.Location = New-Object Drawing.Point(380,160)
$ok.Size = New-Object Drawing.Size(100,30)
$ok.DialogResult = [Windows.Forms.DialogResult]::OK
$cancel = New-Object Windows.Forms.Button
$cancel.Text = '取消'
$cancel.Location = New-Object Drawing.Point(490,160)
$cancel.DialogResult = [Windows.Forms.DialogResult]::Cancel
$form.Controls.AddRange(@($urlLabel,$urlBox,$tokenLabel,$tokenBox,$ok,$cancel))
$form.AcceptButton = $ok
$form.CancelButton = $cancel
$form.Add_Shown({$tokenBox.Select()})
try {
    if ($form.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) { exit }
    $env:LAB_MEMBER_SERVICE_URL = $urlBox.Text.Trim()
    $env:LAB_MEMBER_SERVICE_TOKEN = $tokenBox.Text.Trim()
    $env:LAB_MEMBER_SERVICE_DIRECT = '1'
    if ([string]::IsNullOrWhiteSpace($env:LAB_MEMBER_SERVICE_TOKEN)) { throw '联调令牌不能为空' }
    $tokenBox.Clear()
    $env:PYTHONPATH = "$senderRoot;$senderRoot/runtime/joint-demo-deps"
    $senderPython = Join-Path $senderRoot '.venv/Scripts/python.exe'
    Start-Process -FilePath $senderPython -ArgumentList @((Join-Path $senderRoot 'continuous_handoff.py')) -WorkingDirectory $senderRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $senderRoot 'runtime/continuous-cam2.stdout.log') -RedirectStandardError (Join-Path $senderRoot 'runtime/continuous-cam2.stderr.log') | Out-Null
} finally {
    $form.Dispose()
    Remove-Item Env:LAB_MEMBER_SERVICE_TOKEN -ErrorAction SilentlyContinue
}
