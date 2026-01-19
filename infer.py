import sys
import os
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from render import render_sets

# 🚀 修复 1: 强制允许 NumPy 加载 Pickle 数据
orig_load = np.load
np.load = lambda *a, **k: orig_load(*a, allow_pickle=True, **k)

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
    skip_video=True,
    quiet=False,
    start_frame=0,
    end_frame=-1
):
    original_argv = sys.argv
    try:
        # 构造内部参数供 get_combined_args 使用
        sys.argv = ["render.py"]
        sys.argv.extend(["-s", source_path, "--model_path", model_path, "--configs", configs_path])
        sys.argv.extend(["--iteration", str(iteration), "--batch", str(batch_size)])

        # 确保传入的是 .npy 路径
        npy_path = custom_aud_wav.replace(".wav", ".npy")
        sys.argv.extend(["--custom_aud", npy_path])
        sys.argv.extend(["--custom_wav", custom_aud_wav])

        parser = ArgumentParser(description="Testing script parameters")

        # 🚀 修复 2: 只初始化一次参数对象，避免重复注册冲突
        lp = ModelParams(parser, sentinel=True)
        pp = PipelineParams(parser)
        hp = ModelHiddenParams(parser)

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

        args = get_combined_args(parser)
        if args.configs:
            config = Config.fromfile(args.configs)
            args = merge_hparams(args, config)

        # 🚀 修复 3: 强制覆盖渲染逻辑开关
        args.custom_wav = custom_aud_wav
        args.custom_aud = npy_path
        args.skip_train = True
        args.skip_test = True   # 跳过测试集，防止干扰
        args.skip_video = False  # 必须开启视频渲染
        args.only_infer = True

        safe_state(args.quiet)
        print(f"--- 渲染引擎就绪 ---")
        print(f"音频: {args.custom_wav}")
        print(f"特征: {args.custom_aud}")

        # 使用之前初始化好的 lp, hp, pp 对象进行 extract
        render_sets(
            lp.extract(args),
            hp.extract(args),
            args.iteration,
            pp.extract(args),
            args
        )
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser(description="Inference script entry point")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--aud", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", type=str, default=os.path.join(script_dir, "arguments", "args.py"))

    args = parser.parse_args()

    # 自动寻找最新模型
    pc_path = os.path.join(args.model_path, "point_cloud")
    if args.iteration == -1 and os.path.exists(pc_path):
        iters = [int(d.split('_')[-1]) for d in os.listdir(pc_path) if d.startswith("iteration_")]
        if iters: args.iteration = max(iters)

    inference(
        source_path=args.data_path,
        model_path=args.model_path,
        custom_aud_npy=args.aud.replace(".wav", ".npy"),
        custom_aud_wav=args.aud,
        configs_path=args.configs,
        iteration=args.iteration,
        skip_test=True,
        skip_video=False,
        quiet=False
    )
