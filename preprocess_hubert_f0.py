import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def resolve_device(requested_device):
    if requested_device is None:
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but CUDA is not available")

    device_index = device.index if device.index is not None else 0
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise ValueError(
            f"CUDA device index {device_index} is out of range; found {device_count} device(s)"
        )
    return torch.device(f"cuda:{device_index}")


def resolve_num_processes(requested_processes, device):
    device = torch.device(device)
    if requested_processes < 0:
        raise ValueError("num_processes must be greater than or equal to 0")

    num_processes = requested_processes or (os.cpu_count() or 1)
    if device.type == "cuda" and num_processes > 1:
        logger.warning(
            f"CUDA preprocessing uses one worker on {device}; "
            f"reducing num_processes from {num_processes} to 1 to avoid duplicate models"
        )
        return 1
    return num_processes


def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=device,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        #audio_norm = audio / hps.data.max_wav_value

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor)

def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device):
    device = torch.device(device)
    num_processes = resolve_num_processes(num_processes, device)
    logger.info(f"Using worker processes: {num_processes}")

    if num_processes == 1:
        process_batch(filenames, f0p, diff, mel_extractor, device=device)
        return

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='Number of worker processes (0=auto); CUDA uses one worker per selected device'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    try:
        device = resolve_device(args.device)
        num_processes = resolve_num_processes(args.num_processes, device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))

    if args.use_diff:
        print("use_diff")
        print("Loading Mel Extractor...")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        print("Loaded Mel Extractor.")
    else:
        mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)
