import sys
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from render import render_sets
try:
    from mmengine import Config
except ImportError:
    from mmcv import Config
from utils.params_utils import merge_hparams

def inference(
    source_path,
    model_path,
    custom_aud_npy,
    custom_aud_wav,
    configs_path,
    iteration=1000,
    batch_size=1,
    skip_train=True,
    skip_test=True,
    quiet=False,
    start_frame=0,
    end_frame=-1
):
    """
    Encapsulated inference function for EGSTalker.
    """
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Construct sys.argv for get_combined_args
        sys.argv = ["render.py"]
        sys.argv.extend(["-s", source_path])
        sys.argv.extend(["--model_path", model_path])
        sys.argv.extend(["--configs", configs_path])
        sys.argv.extend(["--iteration", str(iteration)])
        sys.argv.extend(["--batch", str(batch_size)])
        sys.argv.extend(["--custom_aud", custom_aud_npy])
        sys.argv.extend(["--custom_wav", custom_aud_wav])
        sys.argv.extend(["--start_frame", str(start_frame)])
        sys.argv.extend(["--end_frame", str(end_frame)])
        
        if skip_train:
            sys.argv.append("--skip_train")
        if skip_test:
            sys.argv.append("--skip_test")
        if quiet:
            sys.argv.append("--quiet")
            
        # Set up command line argument parser (same as render.py)
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
        
        # Parse arguments
        args = get_combined_args(parser)
        print("Rendering " , args.model_path)
        
        if args.configs:
            config = Config.fromfile(args.configs)
            args = merge_hparams(args, config)
            
        # Initialize system state (RNG)
        safe_state(args.quiet)
        args.only_infer = True
        # print(args)
        
        # Run rendering
        render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    # Example usage
    inference(
        source_path="data/obama",
        model_path="output/obama",
        custom_aud_npy="aud.npy",
        custom_aud_wav="aud.wav",
        configs_path="arguments/args.py",
        iteration=10000
    )
