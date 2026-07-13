# Full-quality paper outputs excluding out-of-Petri artifact track IDs.
$ErrorActionPreference = "Stop"
$csv = "starvation2-2026-04-23_analyzed.csv"
$root = "animations/paper_hq_clean"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$exclude = @("--exclude-track-ids", "1637", "2937", "2354", "2366")
$common = @(
    "--all-animations",
    "--normalization", "max_visible_larvae",
    "--track-window-seconds", "10",
    "--window-size-frames", "180",
    "--hc-window-size-frames", "360",
    "--density-bins", "120",
    "--hc-density-bins", "72",
    "--frame-stride", "1",
    "--fps", "16",
    "--summary-hold-seconds", "5",
    "--append-pooled-summary",
    "--petri-dish",
    "--no-uniform-color-track"
) + $exclude

$runs = @(
    @{ slug = "4d_sated_vs_starved"; ga = "4d_fed"; gb = "4d_starved"; method = "hist"; skip = $false; labels = @("4d_fed:4d Sated", "4d_starved:4d Starved") },
    @{ slug = "4d_sated_vs_starved_kde"; ga = "4d_fed"; gb = "4d_starved"; method = "kde"; skip = $true; labels = @("4d_fed:4d Sated", "4d_starved:4d Starved") },
    @{ slug = "5d_sated_vs_starved"; ga = "5d_fed"; gb = "5d_starved"; method = "hist"; skip = $false; labels = @("5d_fed:5d Sated", "5d_starved:5d Starved") },
    @{ slug = "5d_sated_vs_starved_kde"; ga = "5d_fed"; gb = "5d_starved"; method = "kde"; skip = $true; labels = @("5d_fed:5d Sated", "5d_starved:5d Starved") }
)

foreach ($r in $runs) {
    $out = Join-Path $root $r.slug
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $root ("run_" + $r.slug + ".log")
    $labelArgs = @()
    foreach ($lab in $r.labels) { $labelArgs += @("--group-label", $lab) }
    $argv = @(
        $csv,
        "--output-dir", $out,
        "--compare-subplots",
        "--group-condition-column", "condition",
        "--groups", $r.ga, $r.gb,
        "--compare-output-slug", $r.slug,
        "--density-method", $r.method
    ) + $common + $labelArgs
    if ($r.skip) { $argv += "--no-track" }
    Write-Host "`n=== $($r.slug) ($($r.method)) ===`n" -ForegroundColor Cyan
    & python create_larvae_animations.py @argv 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($r.slug)" }
}

Write-Host "`nAll paper_hq_clean runs complete.`n" -ForegroundColor Green
