import os
import argparse
from pathlib import Path
from disvoice import prosody
from tqdm import tqdm
import torch

from src.dataset import parse_speaker


def get_parser():
    """Get parser for input arguments

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--indir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)

    return parser


def save_stat_fo(audio_files, indir, outdir):
    """Create file with fundamental frequency information.

    Args:
        audio_files (_type_): _description_
        indir (_type_): _description_
        outdir (_type_): _description_
    """
    pros = prosody.Prosody()
    # create an empty dictionary to store the results
    f0_stats = {}
    # create the speaker ids
    id_to_spkr = sorted(set([parse_speaker(f, "_") for f in audio_files]))
    spk_name_to_idx = {spk_name: spk_idx for spk_idx, spk_name in enumerate(id_to_spkr)}
    # loop through the audio files in the directory
    for filename in tqdm(audio_files):
        # load the audio file and compute features
        features = pros.extract_features_file(os.path.join(indir, filename))

        # compute the mean and standard deviation of the f0
        mean_f0 = features[0]
        std_f0 = features[1]

        # store the results in the dictionary
        spk_idx = spk_name_to_idx[parse_speaker(filename, "_")]
        f0_stats[spk_idx] = {"f0_mean": mean_f0, "f0_std": std_f0}

    # save the dictionary to a pickle file
    with open(os.path.join(outdir, "f0_stats.pt"), "wb") as f:
        torch.save(f0_stats, f)


def main(args):
    """Save fundamental frequency statistics for SSL training."""
    print("Initializing Process..")

    audio_files = [
        f for f in os.listdir(args.indir) if os.path.isfile(os.path.join(args.indir, f))
    ]

    save_stat_fo(audio_files, args.indir, args.outdir)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
