import argparse
from data_utils.deepspeech_features.deepspeech_features import conv_audios_to_deepspeech

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)  # can end with .npy
    ap.add_argument("--num_frames", type=int, default=-1)  # -1 => infer
    ap.add_argument("--ds_pb", required=True)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    out_path = args.out[:-4] if args.out.endswith(".npy") else args.out
    num_frames = None if args.num_frames < 0 else args.num_frames

    conv_audios_to_deepspeech(
        audios=[args.audio],
        out_files=[out_path],
        num_frames_info=[num_frames],
        deepspeech_pb_path=args.ds_pb,
        audio_window_size=args.win,
        audio_window_stride=args.stride,
    )

if __name__ == "__main__":
    main()
