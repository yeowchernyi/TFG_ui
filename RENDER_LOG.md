# Rendering Pipeline Fixes & Usage Log

## 1. Overview
This log documents the modifications made to the EGSTalker codebase to resolve rendering issues, specifically `MemoryError` on long sequences (7999 frames) and audio-video synchronization mismatches.

## 2. Key Issues Resolved
1.  **MemoryError**: The original `render.py` stored all rendered frames in a list before saving, causing RAM to overflow for long sequences.
2.  **Early Stopping**: The generated video was shorter than the audio because the data loader didn't loop the reference video frames.
3.  **FFmpeg Warnings**: "Width not divisible by 2" and "swscaler" warnings due to non-contiguous memory layouts.

## 3. Code Modifications

### `render.py`

**Incremental Writing & Memory Management**
*Before:*
```python
    image = []
    gt = []
    # ...
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):
        # ...
        image.append(output["rendered_image_tensor"].cpu())
        gt.append(output["gt_tensor"].cpu())
        
    image_tensor = torch.cat(image,dim=0)[:process_until]
    gt_image_tensor = torch.cat(gt,dim=0)[:process_until]
    
    write_frames_to_video(tensor_to_image(gt_image_tensor),gts_path+f'/gt', use_imageio = True)
    write_frames_to_video(tensor_to_image(image_tensor),render_path+'/renders', use_imageio = True)
```

*After:*
```python
    # Initialize writers lazily
    writer_renders = None
    # ...
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):
        # ...
        # Process rendered images immediately to save memory
        rendered_batch = output["rendered_image_tensor"].cpu()
        # ...
        if writer_renders is None:
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             writer_renders = cv2.VideoWriter(render_video_path, fourcc, fps, (new_w, new_h))
        
        for img in rendered_imgs:
            writer_renders.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        # Explicitly delete tensors to free memory
        del output
        del rendered_batch
        # ...
        torch.cuda.empty_cache()
```

**Chunking Support**
*Before:*
```python
    process_until = len(viewpoint_stack.dataset)
```

*After:*
```python
    # Handle subsetting
    total_frames = len(viewpoint_stack.dataset) if hasattr(viewpoint_stack, 'dataset') else len(viewpoint_stack)
    if end_frame == -1 or end_frame > total_frames:
        end_frame = total_frames
    
    if start_frame > 0 or end_frame < total_frames:
        print(f"Rendering subset: {start_frame} to {end_frame}")
        viewpoint_stack = torch.utils.data.Subset(viewpoint_stack, range(start_frame, end_frame))
```

### `scene/talking_dataset_readers.py`

**Frame Extension**
*Before:*
```python
    if custom_aud:
        auds = aud_features
        # No logic to handle length mismatch
```

*After:*
```python
    if custom_aud:
        auds = aud_features
        # Extend frames to match audio length
        num_audio_frames = auds.shape[0]
        num_video_frames = len(frames)
        if num_audio_frames > num_video_frames:
            print(f"Extending video frames from {num_video_frames} to {num_audio_frames} to match custom audio.")
            repeats = num_audio_frames // num_video_frames + 1
            frames = (frames * repeats)[:num_audio_frames]
```

### `infer.py`

**Wrapper Script**
*New File Created:*
```python
def inference(..., start_frame=0, end_frame=-1):
    # ...
    sys.argv.extend(["--start_frame", str(start_frame)])
    sys.argv.extend(["--end_frame", str(end_frame)])
    # ...
    render_sets(...)
```

## 4. How to Run Rendering

### Option A: Standard Inference (Recommended)
If you have sufficient RAM (32GB+), you can run the standard inference script. It now uses the optimized incremental writer.

```bash
python infer.py
```
*Defaults to iteration 10000 and uses `aud.npy`/`aud.wav`.*

### Option B: Low-Memory Chunked Rendering (For 16GB RAM or less)
If you still encounter memory errors, use the provided PowerShell script. It splits the job into 2000-frame chunks, restarting the python process for each chunk to guarantee memory is cleared.

```powershell
.\run_render_chunks.ps1
```
*This script will:*
1. *Render frames 0-2000, 2000-4000, etc.*
2. *Save intermediate videos.*
3. *Merge them into `final_result.mp4` with audio.*

## 5. Output Location
The final rendered video can be found at:
`output/obama/custom/ours_<iteration>/renders/`
