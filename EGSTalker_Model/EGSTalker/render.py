#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
#import torch.multiprocessing as mp
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_from_batch
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import concurrent.futures
from torch.utils.data import DataLoader
import subprocess
import shlex

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, scene, gaussians, pipeline,audio_dir, batch_size, max_frames=-1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir
    
    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)
    
    viewpoint_stack = scene
    try:
        full_len = len(viewpoint_stack)
    except Exception:
        full_len = len(viewpoint_stack.dataset)
    process_until = full_len
    print("[DBG] render_set full_len:", full_len, "process_until:", process_until)

    if max_frames is not None and max_frames > 0:
        process_until = min(process_until, max_frames)

        # If it's a torch Dataset (supports __getitem__), wrap with Subset for clean truncation
        # If it's already a list, slice it.
        try:
            from torch.utils.data import Subset
            if isinstance(viewpoint_stack, list):
                viewpoint_stack = viewpoint_stack[:process_until]
            else:
                viewpoint_stack = Subset(viewpoint_stack, range(process_until))
        except Exception:
            # Fallback: try slicing (works for list-like objects)
            try:
                viewpoint_stack = viewpoint_stack[:process_until]
            except Exception:
                pass
    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=list)
    
    loader = iter(viewpoint_stack_loader)
    
    if name == "train" :
        process_until = min(process_until, 1000)
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------") 
    print("point nums:",gaussians._xyz.shape[0])
    image = []
    gt = []
    
    iterations = process_until // batch_size
    if process_until % batch_size != 0:
        iterations += 1
    total_time = 0
    
    #render image
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):

        viewpoint_cams = next(loader)
        try:
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                random_color= False, stage='fine',
                                batch_size=batch_size, visualize_attention=False, only_infer=True)
        except:
            break
        total_time += output["inference_time"]
        image.append(output["rendered_image_tensor"].cpu())
        gt.append(output["gt_tensor"].cpu())
        
    image_tensor = torch.cat(image,dim=0)[:process_until]
    gt_image_tensor = torch.cat(gt,dim=0)[:process_until]
    
    print("total frame:",(image_tensor.shape[0]))
    print("FPS:",(torch.cat(image,dim=0).shape[0])/(total_time))
    
    
    #render attention
    loader = iter(viewpoint_stack_loader)
    for idx in range(iterations):

        viewpoint_cams = next(loader)
        try:
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                random_color= False, stage='fine',
                                batch_size=batch_size, visualize_attention=True, only_infer=True) 
        except:
            break
        total_time += output["inference_time"]
    
    if name != 'custom':
        write_frames_to_video(tensor_to_image(gt_image_tensor), gts_path + '/gt', use_imageio=False)

    write_frames_to_video(tensor_to_image(image_tensor), render_path + '/renders', use_imageio=False)
    
    in_mp4 = os.path.join(render_path, "renders.mp4")
    in_wav = os.path.abspath(inf_audio_dir)
    out_mp4 = os.path.join(
        render_path,
        f"{os.path.basename(model_path)}_{name}_{iteration}iter_renders_with_audio.mp4"
    )

    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", in_mp4,
        "-i", in_wav,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_mp4
    ]

    print("[FFMPEG]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    print("[OK] wrote video with audio:", out_mp4)


    #if name != 'custom':
    #    os.remove(f"{gts_path}/gt.mp4")
    #os.remove(f"{render_path}/renders.mp4")
    
    
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train, skip_test, skip_video, batch_size= args.skip_train, args.skip_test, args.skip_video, args.batch
    
    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=args.custom_aud)
        
        gaussians.eval()
        
        if args.custom_aud != '':
            if os.path.exists(args.custom_wav):
                audio_dir = args.custom_wav
            else:
                audio_dir = os.path.join(data_dir, args.custom_wav)
        else:
            audio_dir = ""
        custom_cams = list(scene.getCustomCameras())

        # 目标帧数：按 wav 时长 * 25fps
        import subprocess, math
        wav = os.path.abspath(audio_dir)
        dur = float(subprocess.check_output([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1", wav
        ]).decode().strip())
        target = int(math.ceil(dur * 25))

        if len(custom_cams) < target:
            custom_cams.extend([custom_cams[-1]] * (target - len(custom_cams)))

        print("[DBG] wav:", wav)
        print("[DBG] target frames:", target)
        print("[DBG] len(custom_cams):", len(custom_cams))

        render_set(dataset.model_path, "custom", scene.loaded_iter, custom_cams, gaussians, pipeline, audio_dir, args.batch, args.max_frames)

        #FPS
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud_train.wav")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size, max_frames=args.max_frames)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud_novel.wav")
            render_set(dataset.model_path, "test",iteration, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size, max_frames=args.max_frames)

def write_frames_to_video(frames, path, fps=25, codec='mp4v', use_imageio=False, ext='.mp4'):
    # frames: list of HxWx3 uint8 (RGB) or np arrays
    root, suffix = os.path.splitext(path)
    if suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        out_path = path
    else:
        out_path = path + ext

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if use_imageio:
        import imageio
        imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=None)
        return

    if len(frames) == 0:
        raise RuntimeError("No frames to write.")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open. out_path={out_path}, codec={codec}, fps={fps}, size={(w,h)}"
        )

    for f in frames:
        # OpenCV wants BGR
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[OK] wrote video: {out_path}")

def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor
        
    if normalize:
        image = 255 * image
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image        

            
if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--custom_aud", type=str, default='')
    parser.add_argument("--custom_wav", type=str, default='')
    parser.add_argument("--max_frames", type=int, default=-1)
    # parser.add_argument("--audio_dir", type=str)
    from argparse import Namespace

    cli_args = parser.parse_args()

    cfg_path = os.path.join(cli_args.model_path, "cfg_args")
    print("Looking for config file in", cfg_path)

    if os.path.exists(cfg_path):
        print("Config file found:", cfg_path)
        cfg_text = open(cfg_path, "r").read().strip()
        # cfg_args 是一行：Namespace(...)
        cfg_args = eval(cfg_text, {"Namespace": Namespace})
    else:
        raise FileNotFoundError(f"cfg_args not found: {cfg_path}")

    # 以 cfg_args 为基准
    args = cfg_args

    # 用命令行覆盖“你这次运行必须改变”的字段（路径/批量/自定义音频等）
    args.model_path  = cli_args.model_path
    args.source_path = cli_args.source_path
    args.batch       = cli_args.batch
    args.iteration   = cli_args.iteration
    args.custom_aud  = cli_args.custom_aud
    args.custom_wav  = cli_args.custom_wav
    args.max_frames  = cli_args.max_frames
    args.skip_train  = cli_args.skip_train
    args.skip_test   = cli_args.skip_test
    args.skip_video  = cli_args.skip_video

    # 保险：只推理
    args.only_infer = True
    args.eval = True
    args.quiet = getattr(cli_args, "quiet", False)

    # 如果你命令行传了 sh_degree，就用命令行；否则用 cfg_args 里的
    if getattr(cli_args, "sh_degree", None) is not None:
        args.sh_degree = cli_args.sh_degree

    if getattr(args, "sh_degree", None) is None:
        args.sh_degree = 3

    print("Rendering ", getattr(args, "model_path", None))

    # ---- load & merge configs (optional) ----
#    if args.configs:
#        try:
#            from mmengine.config import Config
#        except Exception:
#            from mmcv import Config

#        from utils.params_utils import merge_hparams

#        print("Before merge source_path:", getattr(args, "source_path", None))
#        config = Config.fromfile(args.configs)
#        args = merge_hparams(args, config)
#        print("After  merge source_path:", args.source_path)
#    print("Rendering ", args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.only_infer = True
    print(args)

    dataset = model.extract(args)
    print("ARGS source_path   :", args.source_path)
    print("DATASET source_path:", getattr(dataset, "source_path", None))

    # Key fix: ensure Scene reads the right path
    dataset.source_path = args.source_path

    render_sets(dataset, hyperparam.extract(args), args.iteration, pipeline.extract(args), args)
