import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_PATH = '/scratch/as20482/ML_Final_Proj/audioset-20k-wds/balanced'
OUTPUT_PATH = '/scratch/as20482/ML_Final_Proj/audioset-20k-wds/Data'
target_sample_rate = 16000
origin_sample_rate = 32000


def load_and_resample_waveform(file_path):
    waveform, original_sample_rate = torchaudio.load(file_path)
    assert original_sample_rate == origin_sample_rate
    resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return resample_transform(waveform)

def pad_or_clip_waveform(waveform, target_length):
    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = (0, target_length - current_length)
        waveform = F.pad(waveform, padding)
    else:
        waveform = waveform[:, :target_length]
    return waveform

def pad_label(label, target_length):
    return label + [-1] * (target_length - len(label))

def load_and_index(split = 'test'):
    wav_files = [f for f in os.listdir(os.path.join(DATA_PATH, split)) if f.endswith('.wav')]
    wav_files.sort()
    return wav_files

def load_label_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data["label"]

def save_16khz_audio(split = 'test'):
    wav_files = load_and_index(split)

    resampled_waveforms = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_and_resample_waveform, os.path.join(DATA_PATH, split, file_path)): file_path for file_path in wav_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Waveforms"):
            resampled_waveforms.append(future.result())
    
    assert len(resampled_waveforms) == len(wav_files)

    max_length = max(waveform.shape[1] for waveform in resampled_waveforms)

    padded_waveforms = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(pad_or_clip_waveform, waveform, max_length): i for i, waveform in enumerate(resampled_waveforms)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Padding Waveforms"):
            padded_waveforms.append(future.result())

    assert len(padded_waveforms) == len(wav_files)

    batch_waveforms_tensor = torch.stack(padded_waveforms)

    torch.save(batch_waveforms_tensor, os.path.join(OUTPUT_PATH, split, 'resampled_waveforms.pt'))

    json_files =[os.path.join(DATA_PATH, split, f.replace(".wav", ".json")) for f in wav_files]

    labels = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_label_from_json, json_file): json_file for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Labels"):
            labels.append(future.result())
    
    max_length = max(len(label) for label in labels)

    padded_labels = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(pad_label, label, max_length): i for i, label in enumerate(labels)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Padding Labels"):
            padded_labels.append(future.result())

    labels_tensor = torch.tensor(padded_labels)
    torch.save(labels_tensor, os.path.join(OUTPUT_PATH, split, 'labels.pt'))




