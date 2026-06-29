param(
    [string]$ApkPath = "android\app\build\outputs\apk\debug\app-debug.apk",
    [string]$Serial = ""
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$apk = Resolve-Path (Join-Path $root $ApkPath)
$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
    throw "adb was not found. Install Android Studio platform-tools and add adb to PATH."
}

$adbArgs = @()
if ($Serial) {
    $adbArgs += @("-s", $Serial)
}

& adb @adbArgs install -r $apk
& adb @adbArgs shell am start -n "com.pocketcv.pdf/.MainActivity"
Write-Host "Installed and launched PocketCV PDF Local."
