param(
    [string]$PptPath = "E:\labcams_data\20260601\alignment_comparison_PS94_PS95_PS.pptx"
)

$ErrorActionPreference = "Stop"

function Add-TextBox {
    param(
        $Slide,
        [string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$Size,
        [bool]$Bold = $false,
        [string]$Color = "222222"
    )
    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Aptos"
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = [int]$Bold
    $r = [Convert]::ToInt32($Color.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($Color.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($Color.Substring(4, 2), 16)
    $shape.TextFrame.TextRange.Font.Color.RGB = $r + ($g * 256) + ($b * 65536)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    return $shape
}

function Add-FittedPicture {
    param($Slide, [string]$Path, [double]$BoxLeft, [double]$BoxTop, [double]$BoxWidth, [double]$BoxHeight)
    if (-not (Test-Path $Path)) {
        Add-TextBox $Slide "Missing image:`r`n$Path" $BoxLeft $BoxTop $BoxWidth 80 10 $false "AA0000" | Out-Null
        return $null
    }
    $pic = $Slide.Shapes.AddPicture($Path, $false, $true, $BoxLeft, $BoxTop, -1, -1)
    $scale = [Math]::Min($BoxWidth / $pic.Width, $BoxHeight / $pic.Height)
    $pic.Width = $pic.Width * $scale
    $pic.Height = $pic.Height * $scale
    $pic.Left = $BoxLeft + (($BoxWidth - $pic.Width) / 2)
    $pic.Top = $BoxTop + (($BoxHeight - $pic.Height) / 2)
    return $pic
}

function Add-Footer {
    param($Slide, [string]$Text)
    Add-TextBox $Slide $Text 32 518 896 14 6.5 $false "666666" | Out-Null
}

function Add-ImageSlide {
    param(
        $Presentation,
        [int]$BlankLayout,
        [string]$Title,
        [string]$Subtitle,
        [string]$ImagePath,
        [string]$FooterPrefix = ""
    )
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, $BlankLayout)
    Add-TextBox $slide $Title 32 18 896 24 19 $true "222222" | Out-Null
    Add-TextBox $slide $Subtitle 32 46 896 17 8.5 $false "666666" | Out-Null
    Add-FittedPicture $slide $ImagePath 28 72 904 435 | Out-Null
    $footer = $ImagePath
    if ($FooterPrefix) {
        $footer = "$FooterPrefix | $ImagePath"
    }
    Add-Footer $slide $footer
}

function Add-SectionSlide {
    param($Presentation, [int]$BlankLayout, [string]$Title, [string]$Subtitle)
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, $BlankLayout)
    $slide.Background.Fill.ForeColor.RGB = 0x111111
    Add-TextBox $slide $Title 46 188 860 54 34 $true "FFFFFF" | Out-Null
    Add-TextBox $slide $Subtitle 50 252 850 60 13 $false "D8D8D8" | Out-Null
    Add-TextBox $slide "PCO_ALIGNED_APPEND_SECTION" 900 520 40 8 1 $false "111111" | Out-Null
}

function Get-FrameQcLine {
    param([string]$SummaryPath, [string]$EventName = "cue")
    if (-not (Test-Path $SummaryPath)) {
        return "Frame QC summary missing"
    }
    $summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $event = $summary.events.$EventName
    if ($null -eq $event) {
        return "Frame QC event '$EventName' missing"
    }
    $raw = $event.raw_frame_delta_stats
    $corr = $event.corrected_frame_delta_stats
    $ms = $event.time_delta_ms_stats
    return "$EventName old-new median: $($raw.median) raw frames, $($corr.median) corrected frames, $([Math]::Round($ms.median, 1)) ms; range $($raw.min) to $($raw.max) raw frames"
}

if (-not (Test-Path $PptPath)) {
    throw "PowerPoint not found: $PptPath"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = [System.IO.Path]::Combine(
    [System.IO.Path]::GetDirectoryName($PptPath),
    "$([System.IO.Path]::GetFileNameWithoutExtension($PptPath))_backup_$timestamp.pptx"
)
Copy-Item -LiteralPath $PptPath -Destination $backupPath -Force

$subjects = @(
    [pscustomobject]@{
        Label = "PS94"
        VersionLabel = "PS94_v4"
        CueFolder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected\spout_trial_averages_allen_v4"
        LickFolder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected\lick_aligned_v4"
        QcFolder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected\frame_alignment_qc"
    },
    [pscustomobject]@{
        Label = "PS95"
        VersionLabel = "PS95_v6"
        CueFolder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6"
        LickFolder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6"
        QcFolder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\frame_alignment_qc"
    }
)

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = 1
$presentation = $ppt.Presentations.Open($PptPath, $false, $false, $true)

try {
    $blankLayout = 12

    # Remove prior generated append section only. This preserves manual edits elsewhere.
    for ($i = $presentation.Slides.Count; $i -ge 1; $i--) {
        $slide = $presentation.Slides.Item($i)
        $delete = $false
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                if ($shape.TextFrame.TextRange.Text -like "*PCO_ALIGNED_APPEND_SECTION*") {
                    $delete = $true
                    break
                }
            }
        }
        if ($delete) {
            $slide.Delete()
        }
    }

    Add-SectionSlide $presentation $blankLayout `
        "PCO-Aligned Shared-Scale Analysis" `
        "Regenerated maps use DAQ-recorded PCO exposure pulse order for event-to-frame alignment. Shared-scale cue panels replace the older separate-scale cue interpretation. Existing manual deck edits were preserved; this section was appended."

    foreach ($subject in $subjects) {
        $summaryPath = Join-Path $subject.QcFolder "$($subject.VersionLabel)_frame_alignment_old_camlog_vs_new_pco_summary.json"
        $qcLine = Get-FrameQcLine $summaryPath "cue"

        Add-SectionSlide $presentation $blankLayout `
            "$($subject.Label) regenerated outputs" `
            "$qcLine`r`nDelta convention in QC: old camlog wall-clock frame index minus new DAQ PCO pulse-order frame index."

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) cue-aligned spout maps - shared scale" `
            "1 s pre-cue, 1 s post-cue, and post-pre panels all use one diverging scale in this figure" `
            (Join-Path $subject.CueFolder "$($subject.VersionLabel)_spout_positions_1s_pre_post_delta_shared_scale.png") `
            "PCO-aligned"

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) cue-aligned pairwise spout-position contrasts" `
            "Each panel is first condition's post-pre map minus second condition's post-pre map; single shared contrast scale" `
            (Join-Path $subject.CueFolder "$($subject.VersionLabel)_pairwise_spout_position_delta_contrasts_allen_overlay.png") `
            "PCO-aligned"

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) post-lick maps by spout position" `
            "150 ms post-lick mean activity; lick detector uses upper/lower hysteresis thresholds; single shared display scale" `
            (Join-Path $subject.LickFolder "$($subject.VersionLabel)_lick_aligned_150ms_post_by_spout.png") `
            "PCO-aligned"

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) post-lick pairwise spout-position contrasts" `
            "Each panel is first condition's post-lick map minus second condition's post-lick map; single shared contrast scale" `
            (Join-Path $subject.LickFolder "$($subject.VersionLabel)_lick_aligned_pairwise_spout_position_contrasts.png") `
            "PCO-aligned"

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) cue-aligned versus lick-aligned maps" `
            "Columns: 1 s post-cue, 150 ms post-lick, and lick minus cue using one shared scale" `
            (Join-Path $subject.LickFolder "$($subject.VersionLabel)_cue_vs_lick_spout_position_maps.png") `
            "PCO-aligned"

        Add-ImageSlide $presentation $blankLayout `
            "$($subject.Label) frame alignment QC: old camlog versus new PCO" `
            "New DAQ PCO pulse-order mapping is treated as hardware reference; histograms show old camlog minus new PCO frame mapping" `
            (Join-Path $subject.QcFolder "$($subject.VersionLabel)_frame_alignment_old_camlog_vs_new_pco.png") `
            "QC"
    }

    $presentation.Save()
    Write-Output "Updated: $PptPath"
    Write-Output "Backup:  $backupPath"
    Write-Output "Slides:  $($presentation.Slides.Count)"
}
finally {
    $presentation.Close()
    $ppt.Quit()
}
