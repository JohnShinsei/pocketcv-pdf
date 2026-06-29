param(
    [string]$Python = "python",
    [string]$Name = "PocketCV-PDF-Local"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $root
try {
    & $Python -m pip install -e ".[api,ocr,desktop]"
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --name $Name `
        --collect-data clearscan_cv `
        --hidden-import uvicorn.logging `
        --hidden-import uvicorn.loops `
        --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.protocols `
        --hidden-import uvicorn.protocols.http `
        --hidden-import uvicorn.protocols.http.auto `
        --hidden-import uvicorn.protocols.websockets `
        --hidden-import uvicorn.lifespan `
        --hidden-import uvicorn.lifespan.on `
        "src\clearscan_cv\local_app.py"
    Write-Host "Built dist\$Name\$Name.exe"
} finally {
    Pop-Location
}
