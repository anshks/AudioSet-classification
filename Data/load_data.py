import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_PATH = '/scratch/as20482/ML_Final_Proj/audioset-20k-wds/balanced'
OUTPUT_PATH = '/scratch/as20482/ML_Final_Proj/AudioSet-classification/Data'
TARGET_SAMPLE_RATE = 16000
ORIGIN_SAMPLE_RATE = 32000
NUM_CLASSES = 527


def load_and_resample_waveform(file_path):
    waveform, original_sample_rate = torchaudio.load(file_path)
    assert original_sample_rate == ORIGIN_SAMPLE_RATE
    resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=TARGET_SAMPLE_RATE)
    return (file_path, resample_transform(waveform))

def pad_or_clip_waveform(waveform, target_length):
    path, waveform = waveform
    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = (0, target_length - current_length)
        waveform = F.pad(waveform, padding)
    else:
        waveform = waveform[:, :target_length]
    return (path, waveform)

def load_and_index(split = 'test'):
    wav_files = [f for f in os.listdir(os.path.join(DATA_PATH, split)) if f.endswith('.wav')]
    return wav_files

def load_label_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return (file_path, data["label"])

def save_16khz_audio(split = 'test'):
    wav_files = load_and_index(split)

    resampled_waveforms = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_and_resample_waveform, os.path.join(DATA_PATH, split, file_path)): file_path for file_path in wav_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Waveforms"):
            resampled_waveforms.append(future.result())
    
    assert len(resampled_waveforms) == len(wav_files)
    
    max_length = max(waveform[1].shape[1] for waveform in resampled_waveforms)

    padded_waveforms = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(pad_or_clip_waveform, waveform, max_length): i for i, waveform in enumerate(resampled_waveforms)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Padding Waveforms"):
            padded_waveforms.append(future.result())

    assert len(padded_waveforms) == len(wav_files)

    padded_waveforms.sort()
    
    batch_waveforms_tensor = torch.stack(list(map(lambda x: x[1], padded_waveforms)))

    torch.save(batch_waveforms_tensor, os.path.join(OUTPUT_PATH, split, 'resampled_waveforms.pt'))

    json_files =[os.path.join(DATA_PATH, split, f.replace(".wav", ".json")) for f in wav_files]

    labels = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_label_from_json, json_file): json_file for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Labels"):
            labels.append(future.result())

    labels.sort()
    
    indices = [torch.tensor(label[1], dtype=torch.long) for label in labels]
    onehot_labels= torch.zeros((len(labels), NUM_CLASSES), dtype=torch.long)
    for i, index in enumerate(indices):
        onehot_labels[i].scatter_(0, index, 1.0)

    torch.save(onehot_labels, os.path.join(OUTPUT_PATH, split, 'labels.pt'))

    assert list(map(lambda x: x[0].replace(".wav", ""), padded_waveforms)) == list(map(lambda x: x[0].replace(".json", ""), labels))




