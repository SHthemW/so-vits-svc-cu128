[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$BuildOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

. (Join-Path $PSScriptRoot "build_executables.ps1")

$RootDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$RootPrefix = $RootDir.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
$IsWindowsPlatform = $env:OS -eq "Windows_NT"
$PlatformCommandFile = if ($IsWindowsPlatform) { "scripts/sovits.bat" } else { "scripts/sovits" }
$BuildRoot = Join-Path $RootDir "build"
$DistDir = Join-Path $RootDir "dist"
$IconPath = Join-Path $RootDir "resource\sovits_ico.ico"
$BuildRequirementsPath = Join-Path $RootDir "requirements_build.txt"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$PackageName = "so-vits-svc-cu128-$Timestamp"
$ArchivePath = Join-Path $DistDir "$PackageName.zip"
$ChecksumPath = "$ArchivePath.sha256"
$ManifestPath = Join-Path ([IO.Path]::GetTempPath()) (
    "so-vits-svc-build-{0}-{1}.txt" -f $Timestamp, [Guid]::NewGuid().ToString("N")
)

$Entrypoints = @(
    [PSCustomObject]@{
        Script = "start_gui.py"
        Name = "so-vits-svc-start_gui"
        Description = "So-VITS-SVC WebUI Launcher"
    },
    [PSCustomObject]@{
        Script = "install_sovits_command.py"
        Name = "so-vits-svc-install_sovits_command"
        Description = "So-VITS-SVC Command Installer"
    }
)

$ExcludedRootDirectories = @(
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".vs",
    ".claude",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
    "shelf",
    "dataset",
    "dataset_raw",
    "raw",
    "results",
    "logs",
    "configs",
    "filelists",
    "checkpoints",
    "trained",
    "output",
    "outputs",
    "downloads",
    "tmp",
    "temp",
    "build",
    "build_exe",
    "dist",
    "_internal",
    "so-vits-svc-start_gui",
    "so-vits-svc-start_gui-internal",
    "so-vits-svc-install_sovits_command",
    "so-vits-svc-install_sovits_command-internal"
)

$EscapedRootDirectories = @(
    $ExcludedRootDirectories | ForEach-Object { [Regex]::Escape($_) }
)
$RootDirectoryPattern = "^(?:" + ($EscapedRootDirectories -join "|") + ")(?:/|$)"
$NestedCachePattern = "(?:^|/)(?:__pycache__|\.ruff_cache|\.pytest_cache|\.mypy_cache|\.cache|\.ipynb_checkpoints)(?:/|$)"
$DevelopmentFilePattern = "^(?:\.gitattributes|\.gitignore|\.ruff\.toml|_build\.bat|requirements_build\.txt|scripts/build_release\.ps1|scripts/build_executables\.ps1)$"
$OtherPlatformFilePattern = if ($IsWindowsPlatform) { "^scripts/sovits$" } else { "^scripts/sovits\.bat$" }
$PrivateFilePattern = "^(?:workspace\.xml|webui_config\.json|webui_start(?:\.err)?\.log|inference/chunks_temp\.json|python_env/pyvenv\.cfg)$"
$SensitiveDirectoryPattern = "(?:^|/)(?:\.ssh|\.aws|\.azure|\.gnupg)(?:/|$)"
$SensitiveFilePattern = "(?:^|/)(?:\.env(?:\..*)?|\.netrc|\.pypirc|\.npmrc|pip\.(?:ini|conf)|credentials(?:\.json)?|client[_-]?secrets?\.json|service[_-]?account\.json|secrets?\.(?:json|ya?ml)|token(?:\.json)?|auth\.json|direct_url\.json|id_(?:rsa|ed25519)(?:\.pem)?|[^/]+\.(?:key|pfx|p12))$"
$GeneratedFilePattern = "(?:^|/)(?:Thumbs\.db|desktop\.ini|\.DS_Store)$|(?:\.py[co]|\.log|\.tmp|\.temp|\.bak|\.swp|\.swo)$|~$"

$RequiredSourceFiles = @(
    "AUTHORS.md",
    "start_gui.py",
    "install_sovits_command.py",
    "launcher_runtime.py",
    $PlatformCommandFile,
    "webUI.py",
    "webui_train.py",
    "resource/sovits_ico.ico",
    "python_env/Python/python.exe",
    "python_env/Scripts/python.exe",
    "pretrain/checkpoint_best_legacy_500.pt",
    "pretrain/rmvpe.pt",
    "pretrain/nsf_hifigan/model"
)

function Normalize-ArchivePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $NormalizedPath = $Path.Replace("\", "/")
    while ($NormalizedPath.StartsWith("./", [StringComparison]::Ordinal)) {
        $NormalizedPath = $NormalizedPath.Substring(2)
    }
    return $NormalizedPath
}

function Test-SensitivePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $NormalizedPath = Normalize-ArchivePath -Path $RelativePath
    return (
        $NormalizedPath -match $SensitiveDirectoryPattern -or
        $NormalizedPath -match $SensitiveFilePattern
    )
}

function Test-ExcludedSourcePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $NormalizedPath = Normalize-ArchivePath -Path $RelativePath
    return (
        $NormalizedPath -match $RootDirectoryPattern -or
        $NormalizedPath -match $NestedCachePattern -or
        $NormalizedPath -match $DevelopmentFilePattern -or
        $NormalizedPath -match $OtherPlatformFilePattern -or
        $NormalizedPath -match $PrivateFilePattern -or
        $NormalizedPath -match $SensitiveDirectoryPattern -or
        $NormalizedPath -match $SensitiveFilePattern -or
        $NormalizedPath -match $GeneratedFilePattern
    )
}

function Test-BuiltBundlePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $NormalizedPath = Normalize-ArchivePath -Path $RelativePath
    $ExecutableSuffix = if ($IsWindowsPlatform) { ".exe" } else { "" }
    foreach ($Entrypoint in $Entrypoints) {
        if ($NormalizedPath -eq "$($Entrypoint.Name)$ExecutableSuffix") {
            return $true
        }
    }
    return $false
}

function ConvertTo-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [IO.FileInfo]$File
    )

    if (-not $File.FullName.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "File is outside the project directory: $($File.FullName)"
    }
    return $File.FullName.Substring($RootPrefix.Length).Replace("\", "/")
}

function Get-ReleaseSourceFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GitExecutable
    )

    $TrackedFiles = @(& $GitExecutable -C $RootDir -c core.quotepath=false ls-files --)
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed with exit code $LASTEXITCODE"
    }

    $TrackedFiles = @(
        $TrackedFiles | Where-Object {
            if ([string]::IsNullOrWhiteSpace($_)) {
                return $false
            }
            $RelativePath = Normalize-ArchivePath -Path $_
            $FullPath = Join-Path $RootDir $RelativePath.Replace("/", "\")
            return Test-Path -LiteralPath $FullPath -PathType Leaf
        }
    )

    $BundledFiles = @("python_env", "pretrain") | ForEach-Object {
        Get-ChildItem -LiteralPath (Join-Path $RootDir $_) -Recurse -Force -File |
            ForEach-Object { ConvertTo-RelativePath -File $_ }
    }

    return @($TrackedFiles + $RequiredSourceFiles + $BundledFiles) |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_) -and
            -not (Test-ExcludedSourcePath -RelativePath $_)
        } |
        ForEach-Object { Normalize-ArchivePath -Path $_ } |
        Sort-Object -Unique
}

function Test-PublishedLaunchers {
    $ExecutableSuffix = if ($IsWindowsPlatform) { ".exe" } else { "" }
    $ExecutablePaths = @()

    foreach ($Entrypoint in $Entrypoints) {
        $ExecutablePath = Join-Path $RootDir "$($Entrypoint.Name)$ExecutableSuffix"
        if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
            throw "Published executable is missing: $ExecutablePath"
        }

        $RelativePath = ConvertTo-RelativePath -File (Get-Item -LiteralPath $ExecutablePath)
        if (Test-SensitivePath -RelativePath $RelativePath) {
            throw "Published executable matches a sensitive path rule: $ExecutablePath"
        }

        $ExecutablePaths += $ExecutablePath
    }

    return $ExecutablePaths
}

function Test-ExclusionRules {
    foreach ($ExcludedPath in @(
        ".env",
        ".gitattributes",
        ".gitignore",
        ".ruff.toml",
        "dataset/speaker/audio.wav",
        "python_env/package/direct_url.json",
        "python_env/package/__pycache__/module.pyc",
        "webui_config.json",
        "_internal/python39.dll",
        "so-vits-svc-start_gui/_internal/python39.dll",
        ".ssh/id_ed25519"
    )) {
        if (-not (Test-ExcludedSourcePath -RelativePath $ExcludedPath)) {
            throw "Release exclusion rule did not reject: $ExcludedPath"
        }
    }

    foreach ($AllowedPath in @("webUI.py", "pretrain/rmvpe.pt", "resource/sovits_ico.ico")) {
        if (Test-ExcludedSourcePath -RelativePath $AllowedPath) {
            throw "Release exclusion rule rejected a required path: $AllowedPath"
        }
    }

    $ExecutableSuffix = if ($IsWindowsPlatform) { ".exe" } else { "" }
    foreach ($Entrypoint in $Entrypoints) {
        $PublishedExecutable = "$($Entrypoint.Name)$ExecutableSuffix"
        if (-not (Test-BuiltBundlePath -RelativePath $PublishedExecutable)) {
            throw "Published executable rule did not accept: $PublishedExecutable"
        }
    }
}

$ArchiveOwned = $false

try {
    Test-ExclusionRules

    if ($DryRun -and $BuildOnly) {
        throw "DryRun and BuildOnly cannot be used together"
    }

    $GitCommand = Get-Command git.exe -ErrorAction Stop
    $TarCommand = Get-Command tar.exe -ErrorAction Stop

    foreach ($RequiredPath in @($IconPath, $BuildRequirementsPath)) {
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "Required build file is missing: $RequiredPath"
        }
    }
    foreach ($Entrypoint in $Entrypoints) {
        $ScriptPath = Join-Path $RootDir $Entrypoint.Script
        if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
            throw "Entrypoint script is missing: $ScriptPath"
        }
    }
    foreach ($RequiredDirectory in @("python_env", "pretrain")) {
        if (-not (Test-Path -LiteralPath (Join-Path $RootDir $RequiredDirectory) -PathType Container)) {
            throw "Required release directory is missing: $RequiredDirectory"
        }
    }

    $Files = @(Get-ReleaseSourceFiles -GitExecutable $GitCommand.Source)
    if ($Files.Count -eq 0) {
        throw "The release file list is empty"
    }

    $MissingFiles = @($RequiredSourceFiles | Where-Object { $_ -notin $Files })
    if ($MissingFiles.Count -gt 0) {
        throw "Required release files are missing: $($MissingFiles -join ', ')"
    }

    Write-Host "[Build] Source files selected: $($Files.Count)"
    if ($DryRun) {
        Write-Host "[Build] Dry run completed successfully; executable generation was skipped."
        Write-Host "[Build] Archive would be written to:"
        Write-Host "        $ArchivePath"
        return
    }

    Build-LauncherExecutables `
        -RootDir $RootDir `
        -BuildRoot $BuildRoot `
        -RequirementsPath $BuildRequirementsPath `
        -IconPath $IconPath `
        -Entrypoints $Entrypoints `
        -IsWindowsPlatform $IsWindowsPlatform

    $ExecutablePaths = @(Test-PublishedLaunchers)
    Write-Host "[Build] Published launchers verified: $($ExecutablePaths.Count)"

    if ($BuildOnly) {
        Write-Host "[Build] Build-only run completed successfully."
        Write-Host "[Build] Launcher directory: $RootDir"
        return
    }

    New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
    if ((Test-Path -LiteralPath $ArchivePath) -or (Test-Path -LiteralPath $ChecksumPath)) {
        throw "Build output already exists: $ArchivePath"
    }

    $BundleFiles = @(
        $ExecutablePaths | ForEach-Object {
            ConvertTo-RelativePath -File (Get-Item -LiteralPath $_)
        }
    )
    $ArchiveFiles = @($Files + $BundleFiles) | Sort-Object -Unique
    [IO.File]::WriteAllLines($ManifestPath, [string[]]$ArchiveFiles, $Utf8NoBom)
    Write-Host "[Build] Creating archive:"
    Write-Host "        $ArchivePath"
    Write-Host "[Build] This can take several minutes because the bundled runtime is large."

    $ArchiveOwned = $true
    & $TarCommand.Source -a -c -f $ArchivePath -C $RootDir -T $ManifestPath
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        throw "tar reported success but did not create the archive"
    }

    $ArchiveEntries = @(& $TarCommand.Source -t -f $ArchivePath) |
        ForEach-Object { Normalize-ArchivePath -Path $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "The archive could not be listed"
    }

    $ForbiddenEntries = @(
        $ArchiveEntries | Where-Object {
            if (Test-BuiltBundlePath -RelativePath $_) {
                return Test-SensitivePath -RelativePath $_
            }
            return Test-ExcludedSourcePath -RelativePath $_
        }
    )
    if ($ForbiddenEntries.Count -gt 0) {
        $Preview = ($ForbiddenEntries | Select-Object -First 5) -join ", "
        throw "The archive contains forbidden entries: $Preview"
    }

    $ExecutableSuffix = if ($IsWindowsPlatform) { ".exe" } else { "" }
    $RequiredArchiveEntries = @($RequiredSourceFiles)
    $RequiredArchiveEntries += @(
        $Entrypoints | ForEach-Object { "$($_.Name)$ExecutableSuffix" }
    )
    $MissingArchiveEntries = @(
        $RequiredArchiveEntries | Where-Object { $_ -notin $ArchiveEntries }
    )
    if ($MissingArchiveEntries.Count -gt 0) {
        throw "The archive is missing required files: $($MissingArchiveEntries -join ', ')"
    }

    Write-Host "[Build] Archive entries verified: $($ArchiveEntries.Count)"
    $Hash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $HashLine = "$Hash  $([IO.Path]::GetFileName($ArchivePath))$([Environment]::NewLine)"
    [IO.File]::WriteAllText($ChecksumPath, $HashLine, $Utf8NoBom)

    $ArchiveSize = (Get-Item -LiteralPath $ArchivePath).Length
    $ArchiveOwned = $false
    Write-Host "[Build] Completed successfully."
    Write-Host "[Build] Archive: $ArchivePath"
    Write-Host "[Build] Size:    $ArchiveSize bytes"
    Write-Host "[Build] SHA256:  $ChecksumPath"
}
catch {
    if ($ArchiveOwned) {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $ChecksumPath -Force -ErrorAction SilentlyContinue
    }

    Write-Error "Build failed: $($_.Exception.Message)"
    exit 1
}
finally {
    Remove-Item -LiteralPath $ManifestPath -Force -ErrorAction SilentlyContinue
}
