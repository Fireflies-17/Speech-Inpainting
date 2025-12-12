import logging
import os
import json
import random
import joblib
import librosa
import torch
import torchaudio
import numpy as np

from tqdm import tqdm
from queue import Queue
from argparse import ArgumentParser
from multiprocessing import Manager
from scipy.io.wavfile import write

from src.model import CodeGenerator
from src.multiseries import match_length
from src.preprocess import normalize_nonzero
from src.dataset import extract_fo, generate
from src.utils import (
    AttrDict,
    get_audio_files,
    get_feature_reader,
    parse_speaker,
    load_checkpoint,
    scan_checkpoint,
    get_logger,
)


def get_parser():
    parser = ArgumentParser(description="Inpainting using SSL methods.")
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default="hubert",
        required=False,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path", type=str, help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--sentence_path",
        type=str,
        default=None,
        help="Sentence file containing the root dir and file sentence",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".flac",
        help="Features file path (.wav, .flac, ...)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for process")
    parser.add_argument(
        "--channel_id",
        choices=["1", "2"],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_files",
        help="The folder to save the generated signals",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Hifi-GAN model file path to use for generation",
    )
    return parser


def init_worker(queue, args, h):
    """Initialise workers for processing

    Args:
        queue (Queue): queue object
        args (_type_): inputs arguments
        h (_type_): config file

    Returns:
        _type_: generator and device
    """
    logging.getLogger().handlers = []

    idx = queue.get()
    # device = idx
    device = torch.device(args.device)

    if os.path.isdir(args.checkpoint_file):
        config_file = os.path.join(args.checkpoint_file, "config.json")
    else:
        config_file = os.path.join(
            os.path.split(args.checkpoint_file)[0], "config.json"
        )
    with open(config_file, "r", encoding="utf-8") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Deprecation
    assert h["f0_normalize"] is True, "Only `f0_normalize==True` is supported."
    # Remove vq loss commit
    h.lambda_commit_code = None
    generator = CodeGenerator(h).to(device)
    if os.path.isdir(args.checkpoint_file):
        cp_g = scan_checkpoint(args.checkpoint_file, "g_")
    else:
        cp_g = args.checkpoint_file
    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g["generator"])

    os.makedirs(args.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return generator, device


def inpainting(
    args,
    h,
    generator,
    reader,
    kmeans_model,
    device,
    emb,
    audio_path: str,
    mask_size: int,
    spk_idx: int,
):
    """Inpainting of audio signals by using SSL model (HuBERT) and generating with
    neural vocoder (Hifi-GAN)

    Args:
        args (_type_): _description_
        h (_type_): _description_
        generator (_type_): _description_
        reader (_type_): _description_
        kmeans_model (_type_): _description_
        device (_type_): _description_
        emb (_type_): _description_
        audio_path (str): _description_
        mask_size (int): _description_
        spk_idx (int): _description_

    Returns:
        _type_: _description_
    """
    audio_gt = reader.read_audio(audio_path, channel_id=args.channel_id)
    y = audio_gt.copy()

    # Create a mask to specify the missing region
    # In this example, we'll create a mask that sets a portion of the audio to zero
    # frame_start = np.random.randint(audio_gt.size // 3, audio_gt.size // 3 * 2)
    frame_start = int(h["sampling_rate"] * 3 / 2)
    mask = np.ones_like(y)
    mask[frame_start : frame_start + mask_size] = 0
    # Apply the mask to the audio to perform inpainting (fill the missing region with zeros)
    y_inpainting = (y + 1e-6) * mask
    audio_mask = y_inpainting.copy()

    # Get the features from HuBERT
    feats = reader.get_feats(None, signal=y, channel_id=args.channel_id)
    feats_inpainting = reader.get_feats(
        None, signal=y_inpainting, channel_id=args.channel_id
    )

    feats = feats.cpu().numpy()
    feats_inpainting = feats_inpainting.cpu().numpy()

    # Get the units from quantified HuBERT
    pred = kmeans_model.predict(feats)
    pred_inpainting = kmeans_model.predict(feats_inpainting)

    code = pred
    code_inpainting = pred_inpainting
    code_inpainting[: (frame_start) // h.code_hop_size] = code[
        : (frame_start) // h.code_hop_size
    ]
    code_inpainting[(frame_start + mask_size) // h.code_hop_size :] = code[
        (frame_start + mask_size) // h.code_hop_size :
    ]

    f0 = extract_fo(audio_gt, h["sampling_rate"])
    fo = normalize_nonzero(f0, np.mean(f0), np.std(f0))
    fo = np.expand_dims(fo, axis=0)
    spk_idx = np.array([spk_idx], dtype=np.int64)
    audio_gt, audio_mask, code, fo = match_length(
        [
            (audio_gt, 1),
            (audio_mask, 1),
            (code, h.code_hop_size),
            (fo, int(h["sampling_rate"] * 0.005)),
        ],
        -1,
    )
    audio_gt = torch.FloatTensor(audio_gt)
    code = {
        "code": torch.LongTensor(code).to(device).unsqueeze(0),
        "f0": torch.FloatTensor(fo).to(device).unsqueeze(0),
        "emb": torch.LongTensor(emb).to(device).unsqueeze(0),
        "spkr": torch.LongTensor(spk_idx).to(device).unsqueeze(0),
    }
    code_inpainting = {
        "code": torch.LongTensor(code_inpainting).to(device).unsqueeze(0),
        "f0": torch.FloatTensor(fo).to(device).unsqueeze(0),
        "emb": torch.LongTensor(emb).to(device).unsqueeze(0),
        "spkr": torch.LongTensor(spk_idx).to(device).unsqueeze(0),
    }

    if h.get("f0_vq_params", None) or h.get("f0_quantizer", None):
        to_remove = audio_gt.shape[-1] % (16 * 80)
        assert to_remove % h["code_hop_size"] == 0

        if to_remove != 0:
            to_remove_code = to_remove // h["code_hop_size"]
            to_remove_f0 = to_remove // 80

            audio_gt = audio_gt[:-to_remove]
            audio_mask = audio_mask[:-to_remove]
            code["code"] = code["code"][..., :-to_remove_code]
            code["f0"] = code["f0"][..., :-to_remove_f0]
            code_inpainting["code"] = code_inpainting["code"][..., :-to_remove_code]
            code_inpainting["f0"] = code_inpainting["f0"][..., :-to_remove_f0]

    audio_gen, _ = generate(h, generator, code)
    audio_inp, _ = generate(h, generator, code_inpainting)

    audio_gen = librosa.util.normalize(audio_gen.astype(np.float32))
    audio_mask = librosa.util.normalize(audio_mask.astype(np.float32))
    audio_inp = librosa.util.normalize(audio_inp.astype(np.float32))
    audio_gt = librosa.util.normalize(audio_gt.squeeze().numpy().astype(np.float32))

    return audio_gt, audio_mask, audio_gen, audio_inp


def main(args, logger):
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    if os.path.isdir(args.checkpoint_file):
        config_file = os.path.join(args.checkpoint_file, "config.json")
    else:
        config_file = os.path.join(
            os.path.split(args.checkpoint_file)[0], "config.json"
        )
    with open(config_file, "r", encoding="utf-8") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Deprecation
    assert h["f0_normalize"] is True, "Only `f0_normalize==True` is supported."

    if os.path.isdir(args.checkpoint_file):
        cp_g = scan_checkpoint(args.checkpoint_file, "g_")
    else:
        cp_g = args.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    # Feature extraction
    logger.info(
        f"Creation of feature extractor {args.feature_type} acoustic features..."
    )
    if args.device == "cuda":
        use_cuda = True
    elif args.device == "cpu":
        use_cuda = False
    feature_reader_cls = get_feature_reader(args.feature_type)
    reader = feature_reader_cls(
        checkpoint_path=args.acoustic_model_path, layer=args.layer, use_cuda=use_cuda
    )

    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False
    kmeans_model._n_threads = 40

    # Get the files to process
    root_dir, fnames, _ = get_audio_files(args.manifest_path)

    ids = list(range(1))

    idQueue = Queue()
    for i in ids:
        idQueue.put(i)
    generator, device = init_worker(idQueue, args, h)

    id_to_spkr = sorted(set([parse_speaker(f, "_") for f in fnames]))
    spk_name_to_idx = {spk_name: spk_idx for spk_idx, spk_name in enumerate(id_to_spkr)}
    wav2mel = torch.jit.load(h["wav2mel_path"])
    embedder = torch.jit.load(h["embedder_path"]).eval()
    for file in tqdm(fnames):
        spk_idx = spk_name_to_idx[parse_speaker(file, "_")]
        gt_file = os.path.join(root_dir, file + args.extension)
        base_fname = os.path.basename(file).rstrip("." + args.extension.lstrip("."))
        fname_out_name = base_fname.rsplit(".")[0]
        wav_tensor, sample_rate = torchaudio.load(os.path.join(root_dir, file))
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb = embedder.embed_utterance(mel_tensor)
        emb = emb.detach().cpu().numpy()
        for i in range(1, 5):
            mask_size = int(i * 0.1 * h["sampling_rate"])
            win_size = int(mask_size / h["sampling_rate"] * 1000)

            audio_gt, audio_mask, audio_gen, audio_inp = inpainting(
                args,
                h,
                generator,
                reader,
                kmeans_model,
                device,
                emb,
                gt_file,
                mask_size,
                spk_idx,
            )

            gt_file = os.path.join(args.output_dir, fname_out_name + "_gt.wav")
            output_file_inpainting = os.path.join(
                args.output_dir,
                fname_out_name + "_inpainted_" + str(win_size) + ".wav",
            )
            output_file_gen = os.path.join(args.output_dir, fname_out_name + "_gen.wav")
            output_file_mask = os.path.join(
                args.output_dir, fname_out_name + "_masked_" + str(win_size) + ".wav"
            )
            write(gt_file, h["sampling_rate"], audio_gt)
            write(output_file_mask, h["sampling_rate"], audio_mask)
            write(output_file_gen, h["sampling_rate"], audio_gen)
            write(output_file_inpainting, h["sampling_rate"], audio_inp)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
