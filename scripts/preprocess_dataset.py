import time
import random
import librosa
import numpy as np
from pathlib import Path
from audioread.ffdec import FFmpegAudioFile
from tqdm.auto import tqdm
from functools import partial

# import multiprocessing as mp
from multiprocessing import Pool

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.signal import from_beatmap


# audio processing constants
N_FFT = 2048
SR = 22050
HOP_LENGTH = (SR // 1000) * 6  # 6ms hop length
N_MELS = 64

A_DIM = 40


def load_audio(audio_file):
    aro = FFmpegAudioFile(audio_file)
    wave, _ = librosa.load(aro, sr=SR, res_type="fft")
    if wave.shape[0] == 0:
        raise ValueError("Empty audio file")

    # spectrogram
    mfcc = librosa.feature.mfcc(
        y=wave,
        sr=SR,
        n_mfcc=A_DIM,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return mfcc


def prepare_map(data_dir, map_file):
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"Failed to load {map_file.name}: {e}")
        return

    # only load osu!std maps
    if bm.mode != 0:
        print(f"Skipping non-std map {map_file.name}")
        return

    af_dir = "_".join(
        [bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)]
    )
    map_dir = data_dir / map_file.parent.name / af_dir
    spec_path = map_dir / "spec.npy"
    map_path = map_dir / f"{map_file.stem}.map.npy"

    if map_path.exists():
        return

    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"Failed to parse {map_file.name}: {e}")
        return

    if spec_path.exists():
        for i in range(5):
            try:
                spec = np.load(spec_path)
                break
            except ValueError:
                # can be raised if file was created but writing hasn't completed
                # just wait a little for the writing to finish
                time.sleep(0.001 * 2**i)
        else:
            # retried 5 times without success, just skip
            print(f"Failed to load audio for {map_file.name}")
    else:
        try:
            spec = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"Failed to load audio for {map_file.name}: {e}")
            return

        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)

    frame_times = (
        librosa.frames_to_time(
            np.arange(spec.shape[-1]),
            sr=SR,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )
        * 1000
    )

    x = from_beatmap(bm, frame_times)
    with open(map_path, "wb") as f:
        np.save(f, x, allow_pickle=False)


if __name__ == "__main__":
    # src_path = Path("D:/Games/osu!/Songs")
    # dst_path = Path("data/")
    src_path = Path("dataset/")
    dst_path = Path("data_v2/")

    src_maps = list(src_path.glob("**/*.osu"))
    random.shuffle(src_maps)

    with Pool(processes=6) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(prepare_map, dst_path), src_maps),
            total=len(src_maps),
        ):
            pass
