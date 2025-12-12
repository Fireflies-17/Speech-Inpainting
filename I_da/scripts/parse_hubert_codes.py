import argparse
import random
from pathlib import Path
from tqdm import tqdm


def get_parser():
    """Get parser for input arguments

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--extension", type=str, required=True, help="Extension type (.flac)"
    )
    parser.add_argument("--min-dur", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tt", type=float, default=0.05)
    parser.add_argument("--cv", type=float, default=0.05)
    parser.add_argument("--ref-train", type=Path)
    parser.add_argument("--ref-val", type=Path)
    parser.add_argument("--ref-test", type=Path)

    return parser


def parse_manifest(manifest):
    audio_files = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == "{":
                sample = eval(line.strip())
                audio_files += [Path(sample["audio"])]
            else:
                audio_files.append(line.rsplit("|")[0])

    return audio_files


def split(args, samples):
    if args.ref_train is not None:
        train_split = parse_manifest(args.ref_train)
        train_split = [x for x in train_split]
        val_split = parse_manifest(args.ref_val)
        val_split = [x for x in val_split]
        test_split = parse_manifest(args.ref_test)
        test_split = [x for x in test_split]
        tt = []
        cv = []
        tr = []

        # parse
        for sample in samples:
            name = Path(sample["audio"]).name.rsplit(".")[0]
            if name in val_split:
                cv += [sample]
            elif name in test_split:
                tt += [sample]
            else:
                tr += [sample]
                # Keep this option to make sure all code are used
                # assert name in train_split
    else:
        # split
        N = len(samples)
        random.shuffle(samples)
        tt = samples[: int(N * args.tt)]
        cv = samples[int(N * args.tt) : int(N * args.tt + N * args.cv)]
        tr = samples[int(N * args.tt + N * args.cv) :]

    return tr, cv, tt


def save(outdir, tr, cv, tt):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f"train.txt", "w") as f:
        f.write("\n".join([str(x) for x in tr]))
    with open(outdir / f"val.txt", "w") as f:
        f.write("\n".join([str(x) for x in cv]))
    with open(outdir / f"test.txt", "w") as f:
        f.write("\n".join([str(x) for x in tt]))


def main(args):

    random.seed(args.seed)

    with open(args.manifest) as f:
        fnames = [l.strip() for l in f.readlines()]
    wav_root = Path(fnames[0])
    fnames = fnames[1:]

    with open(args.codes) as f:
        codes = [l.strip() for l in f.readlines()]

    # parse
    samples = []
    for fname_dur, code in tqdm(zip(fnames, codes)):
        sample = {}
        fname, dur = fname_dur.split("\t")
        if code.find("|") != -1:
            # fname = code.rsplit("|")[0] + "." + postfix
            fname = code.rsplit("|")[0] + args.extension
            index = [idx for idx, s in enumerate(fnames) if fname in s][0]
            dur = fnames[index].split("\t")[-1]
            code = code.rsplit("|")[-1]

        sample["audio"] = str(wav_root / f"{fname}")
        sample["hubert"] = " ".join(code.split(" "))
        sample["duration"] = int(dur) / 16000

        if args.min_dur and sample["duration"] < args.min_dur:
            continue

        samples += [sample]

    tr, cv, tt = split(args, samples)
    save(args.outdir, tr, cv, tt)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
