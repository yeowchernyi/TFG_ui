import os
import argparse

from data_utils.deepspeech_features.deepspeech_features import conv_audios_to_deepspeech

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="input wav path")
    ap.add_argument("--out_npy", required=True, help="output npy path")
    ap.add_argument("--pb", required=True, help="deepspeech frozen graph .pb path")
    ap.add_argument("--num_frames", type=int, default=None, help="optional, usually leave None")
    args = ap.parse_args()

    wav = os.path.abspath(args.wav)
    out_npy = os.path.abspath(args.out_npy)
    pb = os.path.abspath(args.pb)

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    conv_audios_to_deepspeech(
        audios=[wav],
        out_files=[out_npy],
        num_frames_info=[args.num_frames],
        deepspeech_pb_path=pb,
        audio_window_size=16,
        audio_window_stride=1,
    )

    print("[OK] wrote:", out_npy)

if __name__ == "__main__":
    main()
