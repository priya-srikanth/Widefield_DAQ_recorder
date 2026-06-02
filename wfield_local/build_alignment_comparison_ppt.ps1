param(
    [string]$OutputPath = "E:\labcams_data\20260601\alignment_comparison_PS94_PS95.pptx"
)

$ErrorActionPreference = "Stop"

function Get-VersionNumber {
    param([string]$Version)
    if ([string]::IsNullOrWhiteSpace($Version) -or $Version -eq "unversioned") {
        return 2
    }
    if ($Version -match "v(\d+)") {
        return [int]$Matches[1]
    }
    return 999
}

function Get-DisplayVersion {
    param($Item)
    if ($Item.json_version) {
        return [string]$Item.json_version
    }
    if ($Item.aligned_version) {
        return [string]$Item.aligned_version
    }
    return "unversioned"
}

function Get-ImagePath {
    param($Subject, [string]$AlignedVersion)
    $fileVersion = $AlignedVersion
    if ([string]::IsNullOrWhiteSpace($fileVersion)) {
        $fileVersion = "unversioned"
    }
    return Join-Path $Subject.Folder "$($Subject.Label)_$($fileVersion)_mean470_allen_landmark_points.png"
}

function Get-BeforeAfterImagePath {
    param($Subject, [string]$JsonVersion)
    return Join-Path $Subject.BeforeAfterFolder "$($Subject.Label)_$($JsonVersion)_alignment_before_after.png"
}

function Get-SpoutFolder {
    param($Subject, [string]$AlignedVersion)
    if ($AlignedVersion -eq "unversioned") {
        return Join-Path $Subject.MotionFolder "spout_trial_averages_allen"
    }
    return Join-Path $Subject.MotionFolder "spout_trial_averages_allen_$AlignedVersion"
}

function Get-SpoutImagePath {
    param($Subject, [string]$AlignedVersion)
    $folder = Get-SpoutFolder $Subject $AlignedVersion
    $prefix = $Subject.Label
    if ($AlignedVersion -ne "unversioned") {
        $prefix = "$($Subject.Label)_$AlignedVersion"
    }
    return Join-Path $folder "$($prefix)_spout_positions_1s_pre_post_delta_allen_overlay.png"
}

function Get-SharedScaleSpoutImagePath {
    param($Subject, [string]$AlignedVersion)
    $folder = Get-SpoutFolder $Subject $AlignedVersion
    $prefix = $Subject.Label
    if ($AlignedVersion -ne "unversioned") {
        $prefix = "$($Subject.Label)_$AlignedVersion"
    }
    return Join-Path $folder "$($prefix)_spout_positions_1s_pre_post_delta_shared_scale.png"
}

function Get-DeltaContrastImagePath {
    param($Subject, [string]$AlignedVersion)
    $folder = Get-SpoutFolder $Subject $AlignedVersion
    $prefix = $Subject.Label
    if ($AlignedVersion -ne "unversioned") {
        $prefix = "$($Subject.Label)_$AlignedVersion"
    }
    return Join-Path $folder "$($prefix)_pairwise_spout_position_delta_contrasts_allen_overlay.png"
}

function Get-LickFolder {
    param($Subject)
    return Join-Path $Subject.MotionFolder $Subject.LickFolderName
}

function Get-LickPostImagePath {
    param($Subject)
    $folder = Get-LickFolder $Subject
    return Join-Path $folder "$($Subject.LickLabel)_lick_aligned_150ms_post_by_spout.png"
}

function Get-LickContrastImagePath {
    param($Subject)
    $folder = Get-LickFolder $Subject
    return Join-Path $folder "$($Subject.LickLabel)_lick_aligned_pairwise_spout_position_contrasts.png"
}

function Get-CueVsLickImagePath {
    param($Subject)
    $folder = Get-LickFolder $Subject
    return Join-Path $folder "$($Subject.LickLabel)_cue_vs_lick_spout_position_maps.png"
}

function Add-TextBox {
    param($Slide, [string]$Text, [double]$Left, [double]$Top, [double]$Width, [double]$Height, [double]$Size, [bool]$Bold = $false, [string]$Color = "222222")
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

$subjects = @(
    [pscustomobject]@{
        Label = "PS94"
        Folder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected\alignment_landmark_overlays"
        BeforeAfterFolder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected\alignment_before_after"
        MotionFolder = "E:\labcams_data\20260601\PS94_20260601_141614\motion_corrected"
        LickFolderName = "lick_aligned_v4"
        LickLabel = "PS94_v4"
    },
    [pscustomobject]@{
        Label = "PS95"
        Folder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_landmark_overlays"
        BeforeAfterFolder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_before_after"
        MotionFolder = "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected"
        LickFolderName = "lick_aligned_v6"
        LickLabel = "PS95_v6"
    }
)

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = 1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

try {
    $blankLayout = 12

    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
    $slide.Background.Fill.ForeColor.RGB = 0xF4F4F6
    Add-TextBox $slide "PS94 and PS95 Allen Alignment Comparison" 40 145 880 48 27 $true "111111" | Out-Null
    Add-TextBox $slide "Mean 470 nm images with Allen outlines and transformed landmark points" 42 198 840 24 13 $false "444444" | Out-Null
    Add-TextBox $slide "Red x = clicked image landmark after transform`r`nWhite open circle = atlas target landmark" 42 255 520 42 11 $false "333333" | Out-Null
    Add-Footer $slide $OutputPath

    foreach ($subject in $subjects) {
        $summaryPath = Join-Path $subject.Folder "$($subject.Label)_alignment_landmark_points_summary.json"
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $items = @($summary.matched_versions) | Sort-Object @{Expression = { Get-VersionNumber (Get-DisplayVersion $_) }}

        $sectionSlide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
        $sectionSlide.Background.Fill.ForeColor.RGB = 0x111111
        Add-TextBox $sectionSlide $subject.Label 46 205 850 68 44 $true "FFFFFF" | Out-Null
        Add-TextBox $sectionSlide "$($items.Count) matched alignment versions, ordered by landmark JSON version" 50 278 830 24 14 $false "D8D8D8" | Out-Null

        foreach ($item in $items) {
            $displayVersion = Get-DisplayVersion $item
            $alignedVersion = [string]$item.aligned_version
            $beforeAfterPath = Get-BeforeAfterImagePath $subject $displayVersion
            if (Test-Path $beforeAfterPath) {
                $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
                Add-TextBox $slide "$($subject.Label) $displayVersion before/after alignment transform" 32 18 896 24 19 $true "222222" | Out-Null
                Add-TextBox $slide "Before: red X = clicked native-image landmarks. After: red X = clicked landmarks after transform; white circles = atlas target locations." 32 46 896 17 8.5 $false "666666" | Out-Null
                Add-FittedPicture $slide $beforeAfterPath 28 72 904 435 | Out-Null
                Add-Footer $slide $beforeAfterPath
            }

            $imagePath = Get-ImagePath $subject $alignedVersion
            if (-not (Test-Path $imagePath)) {
                continue
            }

            $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
            Add-TextBox $slide "$($subject.Label) $displayVersion" 32 18 896 24 19 $true "222222" | Out-Null
            Add-TextBox $slide "JSON: $(Split-Path $item.json -Leaf) | aligned output: $alignedVersion" 32 46 896 17 8.5 $false "666666" | Out-Null
            Add-FittedPicture $slide $imagePath 52 72 856 435 | Out-Null
            Add-Footer $slide $imagePath

            $spoutPath = Get-SpoutImagePath $subject $alignedVersion
            if (Test-Path $spoutPath) {
                $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
                Add-TextBox $slide "$($subject.Label) $displayVersion spout-position averages" 32 18 896 24 19 $true "222222" | Out-Null
                Add-TextBox $slide "1 s pre-cue, 1 s post-cue, and post-pre delta maps by spout position" 32 46 896 17 8.5 $false "666666" | Out-Null
                Add-FittedPicture $slide $spoutPath 28 72 904 435 | Out-Null
                Add-Footer $slide $spoutPath
            }

            $sharedSpoutPath = Get-SharedScaleSpoutImagePath $subject $alignedVersion
            if (Test-Path $sharedSpoutPath) {
                $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
                Add-TextBox $slide "$($subject.Label) $displayVersion spout-position averages - shared scale" 32 18 896 24 19 $true "222222" | Out-Null
                Add-TextBox $slide "Same diverging color scale across all pre, post, and post-pre panels in this figure" 32 46 896 17 8.5 $false "666666" | Out-Null
                Add-FittedPicture $slide $sharedSpoutPath 28 72 904 435 | Out-Null
                Add-Footer $slide $sharedSpoutPath
            }

            $deltaPath = Get-DeltaContrastImagePath $subject $alignedVersion
            if (Test-Path $deltaPath) {
                $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
                Add-TextBox $slide "$($subject.Label) $displayVersion pairwise delta-position contrasts" 32 18 896 24 19 $true "222222" | Out-Null
                Add-TextBox $slide "Each panel is first condition's post-pre map minus second condition's post-pre map" 32 46 896 17 8.5 $false "666666" | Out-Null
                Add-FittedPicture $slide $deltaPath 28 72 904 435 | Out-Null
                Add-Footer $slide $deltaPath
            }
        }

        $gridPath = Join-Path $subject.Folder "$($subject.Label)_alignment_landmark_points_comparison_grid.png"
        if (Test-Path $gridPath) {
            $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
            Add-TextBox $slide "$($subject.Label) comparison grid" 32 18 896 24 19 $true "222222" | Out-Null
            Add-TextBox $slide "All matched alignment versions side by side" 32 46 896 17 8.5 $false "666666" | Out-Null
            Add-FittedPicture $slide $gridPath 28 72 904 435 | Out-Null
            Add-Footer $slide $gridPath
        }

        $lickPostPath = Get-LickPostImagePath $subject
        if (Test-Path $lickPostPath) {
            $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
            Add-TextBox $slide "$($subject.Label) post-lick maps by spout position" 32 18 896 24 19 $true "222222" | Out-Null
            Add-TextBox $slide "150 ms post-lick mean activity; no pre-lick baseline window" 32 46 896 17 8.5 $false "666666" | Out-Null
            Add-FittedPicture $slide $lickPostPath 65 72 830 435 | Out-Null
            Add-Footer $slide $lickPostPath
        }

        $lickContrastPath = Get-LickContrastImagePath $subject
        if (Test-Path $lickContrastPath) {
            $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
            Add-TextBox $slide "$($subject.Label) post-lick pairwise spout-position contrasts" 32 18 896 24 19 $true "222222" | Out-Null
            Add-TextBox $slide "Each panel is first condition's 150 ms post-lick map minus second condition's map" 32 46 896 17 8.5 $false "666666" | Out-Null
            Add-FittedPicture $slide $lickContrastPath 28 72 904 435 | Out-Null
            Add-Footer $slide $lickContrastPath
        }

        $cueVsLickPath = Get-CueVsLickImagePath $subject
        if (Test-Path $cueVsLickPath) {
            $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, $blankLayout)
            Add-TextBox $slide "$($subject.Label) cue-aligned versus lick-aligned maps" 32 18 896 24 19 $true "222222" | Out-Null
            Add-TextBox $slide "Columns: 1 s post-cue, 150 ms post-lick, and lick minus cue using a shared scale" 32 46 896 17 8.5 $false "666666" | Out-Null
            Add-FittedPicture $slide $cueVsLickPath 28 72 904 435 | Out-Null
            Add-Footer $slide $cueVsLickPath
        }
    }

    if (Test-Path $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    $presentation.SaveAs($OutputPath)
    Write-Output $OutputPath
}
finally {
    $presentation.Close()
    $ppt.Quit()
}
