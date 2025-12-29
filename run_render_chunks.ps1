$total_frames = 7999
$chunk_size = 2000
$model_path = "output/obama"
$iteration = 30000
$source_path = "data/obama"
$config_path = "arguments/args.py"
$custom_aud = "aud.npy"
$custom_wav = "aud.wav"

$render_dir = "$model_path/custom/ours_$iteration/renders"

# 1. Render in chunks
for ($start = 0; $start -lt $total_frames; $start += $chunk_size) {
    $end = $start + $chunk_size
    if ($end -gt $total_frames) { $end = $total_frames }
    
    Write-Host "Rendering chunk: $start to $end"
    
    python render.py -s $source_path --model_path $model_path --configs $config_path --iteration $iteration --batch 1 --custom_aud $custom_aud --custom_wav $custom_wav --skip_train --skip_test --start_frame $start --end_frame $end
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Rendering failed for chunk $start-$end"
        exit 1
    }
    
    # Force garbage collection between runs (OS level)
    [System.GC]::Collect()
}

# 2. Create file list for ffmpeg
$list_file = "$render_dir/file_list.txt"
if (Test-Path $list_file) { Remove-Item $list_file }

# Filter only chunk files (e.g. renders_0_2000.mp4) and sort numerically by start frame
$video_files = Get-ChildItem "$render_dir/renders_*.mp4" | Where-Object { $_.Name -match "renders_\d+_\d+\.mp4" } | Sort-Object { [int]($_.Name -split "_")[1] }

if ($video_files.Count -eq 0) {
    Write-Error "No video chunks found in $render_dir"
    exit 1
}

foreach ($file in $video_files) {
    $line = "file '$($file.Name)'"
    Add-Content $list_file $line
}

# 3. Concatenate videos
$output_video = "$render_dir/renders_merged.mp4"
$final_video = "$render_dir/final_result.mp4"
$audio_file = "$source_path/$custom_wav"

Write-Host "Merging video chunks..."
ffmpeg -f concat -safe 0 -i $list_file -c copy $output_video -y

# 4. Add audio
Write-Host "Adding audio..."
ffmpeg -i $output_video -i $audio_file -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 $final_video -y

Write-Host "Done! Final video saved to $final_video"
