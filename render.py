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
import imageio
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


def render_set(model_path, name, iteration, scene, gaussians, pipeline, audio_dir, batch_size, start_frame=0, end_frame=-1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir

    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)

    viewpoint_stack = scene

    # Handle subsetting
    total_frames = len(viewpoint_stack.dataset) if hasattr(viewpoint_stack, 'dataset') else len(viewpoint_stack)
    if end_frame == -1 or end_frame > total_frames:
        end_frame = total_frames

    if start_frame > 0 or end_frame < total_frames:
        print(f"Rendering subset: {start_frame} to {end_frame}")
        viewpoint_stack = torch.utils.data.Subset(viewpoint_stack, range(start_frame, end_frame))

    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=list)

    loader = iter(viewpoint_stack_loader)

    if name == "train" :
        process_until = 1000
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        process_until = len(viewpoint_stack)
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------")
    print("point nums:",gaussians._xyz.shape[0])

    iterations = process_until // batch_size
    if process_until % batch_size != 0:
        iterations += 1
    total_time = 0

    # Initialize video writers
    fps = 25

    # Append range to filename if subsetting
    suffix = ""
    if start_frame > 0 or end_frame < total_frames:
        suffix = f"_{start_frame}_{end_frame}"

    render_video_path = os.path.join(render_path, f"renders{suffix}.mp4")
    gt_video_path = os.path.join(gts_path, f"gt{suffix}.mp4") if name != 'custom' else None

    writer_renders = None
    writer_gt = None

    import gc

    #render image
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):

        try:
            viewpoint_cams = next(loader)
        except StopIteration:
            break

        try:
            output = render_from_batch(viewpoint_cams, gaussians, pipeline,
                                random_color= False, stage='fine',
                                batch_size=batch_size, visualize_attention=False, only_infer=True)
        except Exception as e:
            print(f"Rendering failed at batch {idx}: {e}")
            break
        total_time += output["inference_time"]

        # Process rendered images immediately to save memory
        rendered_batch = output["rendered_image_tensor"].cpu()
        gt_batch = output["gt_tensor"].cpu()

        rendered_imgs = tensor_to_image(rendered_batch)
        gt_imgs = tensor_to_image(gt_batch)

        # Handle single image case
        if len(rendered_imgs.shape) == 3:
            rendered_imgs = rendered_imgs[None, ...]
        if len(gt_imgs.shape) == 3:
            gt_imgs = gt_imgs[None, ...]

        # Crop to even dimensions
        h, w = rendered_imgs.shape[1:3]
        new_h = h - (h % 2)
        new_w = w - (w % 2)
        if new_h != h or new_w != w:
            rendered_imgs = rendered_imgs[:, :new_h, :new_w, :]
            gt_imgs = gt_imgs[:, :new_h, :new_w, :]

        rendered_imgs = np.ascontiguousarray(rendered_imgs)
        gt_imgs = np.ascontiguousarray(gt_imgs)

        # Initialize writers lazily
        if writer_renders is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer_renders = cv2.VideoWriter(render_video_path, fourcc, fps, (new_w, new_h))
            if gt_video_path:
                writer_gt = cv2.VideoWriter(gt_video_path, fourcc, fps, (new_w, new_h))

        for img in rendered_imgs:
            writer_renders.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if writer_gt:
            for img in gt_imgs:
                writer_gt.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Explicitly delete tensors to free memory
        del output
        del rendered_batch
        del gt_batch
        del rendered_imgs
        del gt_imgs
        del viewpoint_cams
        torch.cuda.empty_cache()

        if idx % 100 == 0:
            gc.collect()

    if writer_renders:
        writer_renders.release()
    if writer_gt:
        writer_gt.release()

    # Skip attention rendering loop as it consumes time and we don't save it
    # loader = iter(viewpoint_stack_loader)
    # for idx in range(iterations): ...

    # Only run ffmpeg if we processed the full video (or handle merging later)
    if start_frame == 0 and end_frame == total_frames:
        if name != 'custom':
            model_name = model_path.split("/")[-2] if len(model_path.split("/")) >= 2 else "/root/autodl-tmp/GaussianTalker/trained_model"
            print("model_path:", model_path)
            cmd = f'ffmpeg -loglevel quiet -y -i {gts_path}/gt.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {gts_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_gt.mov'
            os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/renders.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_renders.mov'
        os.system(cmd)

        if name != 'custom':
            os.remove(f"{gts_path}/gt.mp4")
        os.remove(f"{render_path}/renders.mp4")
    else:
        print(f"Partial render complete. Video saved to {render_video_path}")


def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train, skip_test, skip_video, batch_size= args.skip_train, args.skip_test, args.skip_video, args.batch

    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=args.custom_aud)

        gaussians.eval()

        if args.custom_aud != '':
            audio_dir = os.path.join(data_dir, args.custom_wav)
            render_set(dataset.model_path, "custom", scene.loaded_iter, scene.getCustomCameras(), gaussians, pipeline, audio_dir, batch_size, args.start_frame, args.end_frame)
        #FPS
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud_train.wav")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size, args.start_frame, args.end_frame)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud_novel.wav")
            render_set(dataset.model_path, "test",iteration, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size, args.start_frame, args.end_frame)

def write_frames_to_video(frames, path, codec='mp4v', fps=25, use_imageio=False):
    # Ensure dimensions are even to avoid ffmpeg alignment warnings
    if hasattr(frames, 'shape') and len(frames.shape) == 4:
        h, w = frames.shape[1:3]
        new_h = h - (h % 2)
        new_w = w - (w % 2)
        if new_h != h or new_w != w:
            frames = frames[:, :new_h, :new_w, :]

        # Ensure contiguous memory layout to fix swscaler warning
        frames = np.ascontiguousarray(frames)

    if use_imageio:
        imageio.mimwrite(f'{path}.mp4', frames, fps=fps, quality=8, output_params=['-vf', f'fps={fps}'], macro_block_size=None)
    else:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(f'{path}.mp4', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()

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
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    # parser.add_argument("--audio_dir", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        try:
            from mmengine import Config
        except ImportError:
            from mmcv import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    print(args)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)
