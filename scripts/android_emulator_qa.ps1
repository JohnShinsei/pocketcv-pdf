param(
    [string]$AvdName = "PocketCV_API35",
    [string]$Serial = "",
    [string]$ApkPath = "android\app\build\outputs\apk\debug\app-debug.apk",
    [int]$BackendPort = 8765,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutDir = Join-Path $Root "tmp\android-qa"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-SdkPath {
    if ($env:ANDROID_HOME -and (Test-Path $env:ANDROID_HOME)) {
        return $env:ANDROID_HOME
    }
    $fallback = Join-Path $env:LOCALAPPDATA "Android\Sdk"
    if (Test-Path $fallback) {
        return $fallback
    }
    throw "Android SDK was not found. Set ANDROID_HOME or install the SDK first."
}

function Get-AndroidTool([string]$RelativePath) {
    $tool = Join-Path (Get-SdkPath) $RelativePath
    if (-not (Test-Path $tool)) {
        throw "Missing Android tool: $tool"
    }
    return $tool
}

function Invoke-CmdLine([string]$CommandLine) {
    & cmd.exe /d /s /c $CommandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
    }
}

function Wait-HttpHealth([string]$Url, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $health = Invoke-RestMethod -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($health.status -eq "ok") {
                return $health
            }
        } catch {
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "Backend did not become healthy: $Url"
}

function Start-Backend {
    $url = "http://127.0.0.1:$BackendPort/api/health"
    try {
        return Wait-HttpHealth $url 2
    } catch {
    }

    $backendLog = Join-Path $OutDir "backend.log"
    $src = Join-Path $Root "src"
    $cmd = "set `"PYTHONPATH=$src`" && python -m uvicorn clearscan_cv.api:app --host 127.0.0.1 --port $BackendPort > `"$backendLog`" 2>&1"
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList @("/d", "/s", "/c", $cmd) -WindowStyle Hidden -PassThru
    Write-Host "Started backend wrapper process: $($process.Id)"
    return Wait-HttpHealth $url 45
}

function Dump-Ui([string]$Name) {
    $adb = Get-AndroidTool "platform-tools\adb.exe"
    $path = Join-Path $OutDir $Name
    $cmd = "`"$adb`" -s $Serial exec-out uiautomator dump /dev/tty > `"$path`""
    Invoke-CmdLine $cmd
    return $path
}

function Read-UiXml([string]$Path) {
    $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    $start = $text.IndexOf("<hierarchy")
    $end = $text.IndexOf("</hierarchy>")
    if ($start -lt 0 -or $end -lt 0) {
        throw "UI hierarchy was not found in $Path"
    }
    $xmlText = $text.Substring($start, $end + "</hierarchy>".Length - $start)
    return [xml]$xmlText
}

function Get-BoundsCenter([string]$Bounds) {
    if ($Bounds -notmatch "\[(\d+),(\d+)\]\[(\d+),(\d+)\]") {
        throw "Invalid bounds: $Bounds"
    }
    $x = ([int]$Matches[1] + [int]$Matches[3]) / 2
    $y = ([int]$Matches[2] + [int]$Matches[4]) / 2
    return @([int]$x, [int]$y)
}

function Find-NodeCenter {
    param(
        [string]$UiPath,
        [string]$Text = "",
        [string]$DescriptionPrefix = "",
        [string]$Contains = ""
    )
    $doc = Read-UiXml $UiPath
    foreach ($node in $doc.SelectNodes("//*[@bounds]")) {
        $nodeText = $node.GetAttribute("text")
        $desc = $node.GetAttribute("content-desc")
        $matched = $false
        if ($Text -and $nodeText -eq $Text) {
            $matched = $true
        }
        if ($DescriptionPrefix -and $desc.StartsWith($DescriptionPrefix)) {
            $matched = $true
        }
        if ($Contains -and (($nodeText -like "*$Contains*") -or ($desc -like "*$Contains*"))) {
            $matched = $true
        }
        if ($matched) {
            return Get-BoundsCenter $node.GetAttribute("bounds")
        }
    }
    throw "UI node was not found. text='$Text' descPrefix='$DescriptionPrefix' contains='$Contains'"
}

function Tap-UiNode([string]$UiPath, [string]$Text = "", [string]$DescriptionPrefix = "", [string]$Contains = "") {
    $adb = Get-AndroidTool "platform-tools\adb.exe"
    $point = Find-NodeCenter -UiPath $UiPath -Text $Text -DescriptionPrefix $DescriptionPrefix -Contains $Contains
    & $adb -s $Serial shell input tap $point[0] $point[1]
}

function Wait-UiContains([string]$Needle, [int]$TimeoutSeconds, [string]$FinalDumpName) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Seconds 2
        $ui = Dump-Ui $FinalDumpName
        $text = [System.IO.File]::ReadAllText($ui, [System.Text.Encoding]::UTF8)
        if ($text.Contains($Needle)) {
            return $ui
        }
        if ($text.Contains("失敗:")) {
            throw "App reported failure while waiting for '$Needle'. See $ui"
        }
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for UI text: $Needle"
}

function Ensure-EmulatorOnline {
    $adb = Get-AndroidTool "platform-tools\adb.exe"
    if ($Serial) {
        return
    }

    $existing = & $adb devices | Where-Object { $_ -match "^(emulator-\d+)\s+device$" } | Select-Object -First 1
    if ($existing) {
        $script:Serial = ($existing -split "\s+")[0]
        return
    }

    $emulator = Get-AndroidTool "emulator\emulator.exe"
    $args = @("-avd", $AvdName, "-no-window", "-no-audio", "-no-boot-anim", "-gpu", "swiftshader_indirect", "-no-snapshot")
    $process = Start-Process -FilePath $emulator -ArgumentList $args -WindowStyle Hidden -PassThru
    Write-Host "Started emulator process: $($process.Id)"

    $deadline = (Get-Date).AddMinutes(6)
    do {
        Start-Sleep -Seconds 5
        $line = & $adb devices | Where-Object { $_ -match "^(emulator-\d+)\s+device$" } | Select-Object -First 1
        if ($line) {
            $script:Serial = ($line -split "\s+")[0]
            break
        }
    } while ((Get-Date) -lt $deadline)
    if (-not $Serial) {
        throw "Emulator did not become online. Check AVD '$AvdName'."
    }

    $bootDeadline = (Get-Date).AddMinutes(4)
    do {
        Start-Sleep -Seconds 5
        $boot = & $adb -s $Serial shell getprop sys.boot_completed 2>$null
    } while ($boot.Trim() -ne "1" -and (Get-Date) -lt $bootDeadline)
    if ($boot.Trim() -ne "1") {
        throw "Emulator came online but boot did not complete."
    }
}

$sdk = Get-SdkPath
$env:ANDROID_HOME = $sdk
$env:ANDROID_SDK_ROOT = $sdk
$env:Path = $env:Path + ";" + (Join-Path $sdk "platform-tools") + ";" + (Join-Path $sdk "cmdline-tools\latest\bin")

if (-not $SkipBuild) {
    Push-Location (Join-Path $Root "android")
    try {
        & .\gradlew.bat :app:assembleDebug --no-daemon --stacktrace
        if ($LASTEXITCODE -ne 0) {
            throw "Gradle build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$health = Start-Backend
Write-Host "Backend OK: $($health.app)"

Ensure-EmulatorOnline
Write-Host "Using emulator serial: $Serial"

$adb = Get-AndroidTool "platform-tools\adb.exe"
$apk = Resolve-Path (Join-Path $Root $ApkPath)
& $adb -s $Serial install -r $apk
if ($LASTEXITCODE -ne 0) {
    throw "adb install failed with exit code $LASTEXITCODE"
}
& $adb -s $Serial shell am start -n "com.pocketcv.pdf/.MainActivity"
if ($LASTEXITCODE -ne 0) {
    throw "adb launch failed with exit code $LASTEXITCODE"
}
Start-Sleep -Seconds 2

$mainUi = Dump-Ui "ui-main.xml"
Tap-UiNode -UiPath $mainUi -Text "API確認"
$healthUi = Wait-UiContains "API OK" 30 "ui-after-health.xml"
Write-Host "Android app reached API OK."

& python (Join-Path $Root "scripts\generate_sample.py")
$sample = Resolve-Path (Join-Path $Root "examples\generated\sample_document.jpg")
& $adb -s $Serial push $sample "/sdcard/Download/pocketcv-sample.jpg" | Out-Host
& $adb -s $Serial shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/pocketcv-sample.jpg | Out-Host

Tap-UiNode -UiPath $healthUi -Text "画像を選択"
Start-Sleep -Seconds 4
$pickerUi = Dump-Ui "ui-picker.xml"
Tap-UiNode -UiPath $pickerUi -DescriptionPrefix "pocketcv-sample.jpg"
Start-Sleep -Seconds 3
$selectedUi = Dump-Ui "ui-selected.xml"
Tap-UiNode -UiPath $selectedUi -Text "ローカル後端でスキャン生成"
$processedUi = Wait-UiContains "完了" 60 "ui-processed.xml"
Write-Host "Android app processed the sample image."

$screenshotOnDevice = "/sdcard/Download/pocketcv-processed.png"
$screenshot = Join-Path $OutDir "pocketcv-processed.png"
& $adb -s $Serial shell screencap -p $screenshotOnDevice
& $adb -s $Serial pull $screenshotOnDevice $screenshot | Out-Host

Write-Host "QA artifacts:"
Write-Host "  $processedUi"
Write-Host "  $screenshot"
