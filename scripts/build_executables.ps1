function Resolve-BootstrapPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootDir,

        [Parameter(Mandatory = $true)]
        [bool]$IsWindowsPlatform
    )

    if ($env:SOVITS_BUILD_PYTHON) {
        $ConfiguredPython = [IO.Path]::GetFullPath($env:SOVITS_BUILD_PYTHON)
        if (-not (Test-Path -LiteralPath $ConfiguredPython -PathType Leaf)) {
            throw "SOVITS_BUILD_PYTHON does not point to a file: $ConfiguredPython"
        }
        return $ConfiguredPython
    }

    $BundledCandidates = if ($IsWindowsPlatform) {
        @(
            (Join-Path $RootDir "python_env\Python\python.exe"),
            (Join-Path $RootDir "python_env\Scripts\python.exe")
        )
    }
    else {
        @(
            (Join-Path $RootDir "python_env/bin/python"),
            (Join-Path $RootDir "python_env/bin/python3")
        )
    }

    foreach ($Candidate in $BundledCandidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return $Candidate
        }
    }

    $CommandNames = if ($IsWindowsPlatform) { @("python.exe", "python3.exe") } else { @("python3", "python") }
    foreach ($CommandName in $CommandNames) {
        $Command = Get-Command $CommandName -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }

    throw "No Python interpreter is available for the build"
}


function Get-BuildEnvironmentPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentRoot,

        [Parameter(Mandatory = $true)]
        [bool]$IsWindowsPlatform
    )

    if ($IsWindowsPlatform) {
        return Join-Path $EnvironmentRoot "Scripts\python.exe"
    }
    return Join-Path $EnvironmentRoot "bin/python"
}


function Invoke-CheckedNativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    $ExitCode = 1
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "$FailureMessage (exit code $ExitCode)"
    }
}


function Initialize-BuildEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootDir,

        [Parameter(Mandatory = $true)]
        [string]$EnvironmentRoot,

        [Parameter(Mandatory = $true)]
        [string]$RequirementsPath,

        [Parameter(Mandatory = $true)]
        [bool]$IsWindowsPlatform
    )

    $BuildPython = Get-BuildEnvironmentPython -EnvironmentRoot $EnvironmentRoot -IsWindowsPlatform $IsWindowsPlatform
    $RequirementsStampPath = Join-Path $EnvironmentRoot ".requirements.sha256"
    if (-not (Test-Path -LiteralPath $BuildPython -PathType Leaf)) {
        $BootstrapPython = Resolve-BootstrapPython -RootDir $RootDir -IsWindowsPlatform $IsWindowsPlatform
        Write-Host "[Build] Creating isolated PyInstaller environment with: $BootstrapPython"
        Invoke-CheckedNativeCommand -Executable $BootstrapPython -Arguments @(
            "-m", "venv", $EnvironmentRoot
        ) -FailureMessage "Failed to create the PyInstaller environment"
    }

    $RequirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $InstalledRequirementsHash = if (Test-Path -LiteralPath $RequirementsStampPath -PathType Leaf) {
        [IO.File]::ReadAllText($RequirementsStampPath).Trim().ToLowerInvariant()
    }
    else {
        ""
    }

    $DependenciesAvailable = $false
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $BuildPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('PyInstaller') and importlib.util.find_spec('PIL') else 1)" >$null 2>&1
        $DependenciesAvailable = $LASTEXITCODE -eq 0 -and $InstalledRequirementsHash -eq $RequirementsHash
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if (-not $DependenciesAvailable) {
        Write-Host "[Build] Installing isolated build dependencies..."
        Invoke-CheckedNativeCommand -Executable $BuildPython -Arguments @(
            "-m", "pip", "install", "--disable-pip-version-check", "--requirement", $RequirementsPath
        ) -FailureMessage "Failed to install the build dependencies"
        [IO.File]::WriteAllText($RequirementsStampPath, "$RequirementsHash`n")
    }

    $PyInstallerVersion = & $BuildPython -c "import PyInstaller; print(PyInstaller.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller could not be loaded from the isolated build environment"
    }
    Write-Host "[Build] PyInstaller: $PyInstallerVersion"
    return $BuildPython
}


function Remove-SafeBuildDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$BuildRoot
    )

    $ResolvedBuildRoot = [IO.Path]::GetFullPath($BuildRoot).TrimEnd("\", "/")
    $ResolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    $RequiredPrefix = $ResolvedBuildRoot + [IO.Path]::DirectorySeparatorChar

    if (-not $ResolvedPath.StartsWith($RequiredPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a directory outside the build root: $ResolvedPath"
    }

    if (Test-Path -LiteralPath $ResolvedPath) {
        Remove-Item -LiteralPath $ResolvedPath -Recurse -Force
    }
}


function New-WindowsVersionFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$ExecutableName,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $Content = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [
          StringStruct(u'CompanyName', u'SHW / SHthemW@GitHub'),
          StringStruct(u'Comments', u'Maintained at https://github.com/SHthemW/so-vits-svc-cu128'),
          StringStruct(u'FileDescription', u'$Description'),
          StringStruct(u'FileVersion', u'1.0.0.0'),
          StringStruct(u'InternalName', u'$ExecutableName'),
          StringStruct(u'LegalCopyright', u'Copyright (C) 2026 SHW / SHthemW and So-VITS-SVC contributors'),
          StringStruct(u'OriginalFilename', u'$ExecutableName.exe'),
          StringStruct(u'ProductName', u'So-VITS-SVC CUDA 12.8'),
          StringStruct(u'ProductVersion', u'1.0.0.0')
        ]
      )
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"@

    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}


function Invoke-LauncherCodeSigning {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ExecutablePaths,

        [Parameter(Mandatory = $true)]
        [bool]$IsWindowsPlatform
    )

    $CertificateThumbprint = ($env:SOVITS_SIGN_CERT_SHA1 -replace "\s", "")
    if (-not $CertificateThumbprint) {
        Write-Host "[Build] Code signing skipped; set SOVITS_SIGN_CERT_SHA1 to sign and further reduce antivirus false positives."
        return
    }

    if (-not $IsWindowsPlatform) {
        throw "SOVITS_SIGN_CERT_SHA1 is only supported for Windows builds"
    }
    if ($CertificateThumbprint -notmatch "^[0-9A-Fa-f]{40}$") {
        throw "SOVITS_SIGN_CERT_SHA1 must be a 40-character certificate thumbprint"
    }

    $SignToolPath = $env:SOVITS_SIGNTOOL
    if (-not $SignToolPath) {
        $SignTool = Get-Command signtool.exe -ErrorAction SilentlyContinue
        if ($SignTool) {
            $SignToolPath = $SignTool.Source
        }
    }
    if (-not $SignToolPath -or -not (Test-Path -LiteralPath $SignToolPath -PathType Leaf)) {
        throw "signtool.exe is required when SOVITS_SIGN_CERT_SHA1 is set"
    }

    $TimestampUrl = if ($env:SOVITS_TIMESTAMP_URL) {
        $env:SOVITS_TIMESTAMP_URL
    }
    else {
        "http://timestamp.digicert.com"
    }

    foreach ($ExecutablePath in $ExecutablePaths) {
        Invoke-CheckedNativeCommand -Executable $SignToolPath -Arguments @(
            "sign",
            "/sha1", $CertificateThumbprint,
            "/fd", "SHA256",
            "/td", "SHA256",
            "/tr", $TimestampUrl,
            $ExecutablePath
        ) -FailureMessage "Failed to sign $ExecutablePath"
    }

    Write-Host "[Build] Authenticode signing completed."
}


function Build-LauncherExecutables {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootDir,

        [Parameter(Mandatory = $true)]
        [string]$BuildRoot,

        [Parameter(Mandatory = $true)]
        [string]$RequirementsPath,

        [Parameter(Mandatory = $true)]
        [string]$IconPath,

        [Parameter(Mandatory = $true)]
        [object[]]$Entrypoints,

        [Parameter(Mandatory = $true)]
        [bool]$IsWindowsPlatform
    )

    $EnvironmentRoot = Join-Path $BuildRoot "pyinstaller-env"
    $ExecutableRoot = Join-Path $BuildRoot "launchers"
    $WorkRoot = Join-Path $BuildRoot "pyinstaller-work"
    $SpecRoot = Join-Path $BuildRoot "pyinstaller-spec"
    $VersionRoot = Join-Path $BuildRoot "version-info"
    $ConfigRoot = Join-Path $BuildRoot "pyinstaller-config"

    $BuildPython = Initialize-BuildEnvironment `
        -RootDir $RootDir `
        -EnvironmentRoot $EnvironmentRoot `
        -RequirementsPath $RequirementsPath `
        -IsWindowsPlatform $IsWindowsPlatform

    foreach ($Path in @($ExecutableRoot, $WorkRoot, $SpecRoot, $VersionRoot, $ConfigRoot)) {
        Remove-SafeBuildDirectory -Path $Path -BuildRoot $BuildRoot
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }

    $PreviousConfigRoot = $env:PYINSTALLER_CONFIG_DIR
    $env:PYINSTALLER_CONFIG_DIR = $ConfigRoot

    try {
        foreach ($Entrypoint in $Entrypoints) {
            $ScriptPath = Join-Path $RootDir $Entrypoint.Script
            $EntrypointWorkRoot = Join-Path $WorkRoot $Entrypoint.Name
            $Arguments = @(
                "-m", "PyInstaller",
                "--noconfirm",
                "--clean",
                "--onedir",
                "--noupx",
                "--console",
                "--log-level", "WARN",
                "--name", $Entrypoint.Name,
                "--icon", $IconPath,
                "--distpath", $ExecutableRoot,
                "--workpath", $EntrypointWorkRoot,
                "--specpath", $SpecRoot,
                "--paths", $RootDir
            )

            if ($IsWindowsPlatform) {
                $VersionPath = Join-Path $VersionRoot "$($Entrypoint.Name).txt"
                New-WindowsVersionFile `
                    -Path $VersionPath `
                    -ExecutableName $Entrypoint.Name `
                    -Description $Entrypoint.Description
                $Arguments += @("--version-file", $VersionPath)
            }

            $Arguments += $ScriptPath
            Write-Host "[Build] Building $($Entrypoint.Name) in directory mode with UPX disabled..."
            Invoke-CheckedNativeCommand `
                -Executable $BuildPython `
                -Arguments $Arguments `
                -FailureMessage "PyInstaller failed for $($Entrypoint.Script)"
        }
    }
    finally {
        $env:PYINSTALLER_CONFIG_DIR = $PreviousConfigRoot
    }

    $ExecutableSuffix = if ($IsWindowsPlatform) { ".exe" } else { "" }
    $ExecutablePaths = @()
    foreach ($Entrypoint in $Entrypoints) {
        $ExecutablePath = Join-Path (Join-Path $ExecutableRoot $Entrypoint.Name) "$($Entrypoint.Name)$ExecutableSuffix"
        if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
            throw "Built executable is missing: $ExecutablePath"
        }
        $ExecutablePaths += $ExecutablePath
    }

    Invoke-LauncherCodeSigning -ExecutablePaths $ExecutablePaths -IsWindowsPlatform $IsWindowsPlatform

    foreach ($ExecutablePath in $ExecutablePaths) {
        Invoke-CheckedNativeCommand `
            -Executable $ExecutablePath `
            -Arguments @("--self-test") `
            -FailureMessage "Executable self-test failed for $ExecutablePath"
        Write-Host "[Build] Executable ready: $ExecutablePath"
    }
}
