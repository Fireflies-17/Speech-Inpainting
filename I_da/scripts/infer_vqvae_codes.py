import argparse
import json
import os
import random
from multiprocessing import Manager
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

from src.utils import load_checkpoint, scan_checkpoint
from src.dataset import generate
from src.utils import AttrDict
from src.model import CodeGenerator


def get_parser():
    """Get parser for input arguments

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav_dir", type=Path, required=True)
    parser.add_argument("--output_code_dir", type=Path, required=True)
    parser.add_argument("--output_wav_dir", type=Path, required=False)
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("-n", type=int, default=-1)
    parser.add_argument("--ext", type=str, default="wav")

    return parser


def init_worker(queue, arguments):
    import logging

    logging.getLogger().handlers = []
    global generator
    global encoder
    global vq
    global dataset
    global idx
    global device
    global a
    global h

    args = arguments
    idx = queue.get()
    device = idx

    if os.path.isdir(args.checkpoint_file):
        config_file = os.path.join(args.checkpoint_file, "config.json")
    else:
        config_file = os.path.join(
            os.path.split(args.checkpoint_file)[0], "config.json"
        )
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(args.checkpoint_file):
        cp_g = scan_checkpoint(args.checkpoint_file, "g_")
    else:
        cp_g = args.checkpoint_file
    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    encoder = generator.code_encoder
    encoder.eval()

    vq = generator.code_vq
    vq.eval()

    # fix seed
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def inference(path, args):
    # total_rtf = 0.0
    audio, _ = sf.read(path)
    audio = torch.from_numpy(audio).view(1, 1, -1)
    audio = audio.to(device).float()
    hidden = encoder(audio)
    code, _, _, _ = vq(hidden)
    code = code[0].cpu().squeeze()
    code = ",".join([str(x.item()) for x in code])

    if args.output_wav_dir:
        fname_out_name = Path(path).stem
        new_code = {"code": audio}
        new_code = dict(new_code)
        audio, _ = generate(h, generator, new_code)
        output_file = os.path.join(args.output_wav_dir, fname_out_name + "_gen.wav")
        audio = librosa.util.normalize(audio.astype(np.float32))
        write(output_file, h["sampling_rate"], audio)
    return str(path), code


def main(args):
    """Extract codes from VQVAE process."""

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    files = args.input_wav_dir.glob(f"**/*{args.ext}")
    files = list(files)
    lines = []

    ids = list(range(1))
    import queue

    idQueue = queue.Queue()
    for i in ids:
        idQueue.put(i)
    init_worker(idQueue, args)

    for i, p in enumerate(tqdm(files)):
        l = inference(p, args)
        lines += [l]
        if args.n != -1 and i > args.n:
            break

    args.output_code_dir.mkdir(exist_ok=True)
    with open(args.output_code_dir / "vqvae_output.txt", "w") as f:
        f.write("\n".join("\t".join(l) for l in lines))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
