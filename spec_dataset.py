# spec_dataset.py

import torch
import torchaudio
import os
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T


def load_wav(full_path, target_sr):
    """Loads audio, normalizes it, and ensures it is mono."""
    data, sampling_rate = torchaudio.load(full_path, normalize=True)
    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        data = resampler(data)
    if data.shape[0] > 1:
        data = torch.mean(data, dim=0, keepdim=True)
    return data, target_sr

def get_dataset_filelist(filelist_path):
    """Reads the filelist and returns a list of tuples (clean_path, noisy_path)."""
    with open(filelist_path, 'r', encoding='utf-8') as f:
        filepaths = [line.strip().split('|') for line in f.readlines() if '|' in line]
    return filepaths

def custom_collate_fn(batch):
    """
    Groups and pads spectrograms of different sizes.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        # Return empty batch if all items failed
        return torch.empty(0), torch.empty(0)

    noisy_specs, clean_specs = zip(*batch)

    # Transpose to (Time, Frequency) for padding
    noisy_specs_transposed = [s.T for s in noisy_specs]
    clean_specs_transposed = [s.T for s in clean_specs]

    # Apply padding
    noisy_specs_padded = pad_sequence(noisy_specs_transposed, batch_first=True, padding_value=0.0)
    clean_specs_padded = pad_sequence(clean_specs_transposed, batch_first=True, padding_value=0.0)

    # Transpose back to expected format (Batch, Frequency, Time)
    return noisy_specs_padded.permute(0, 2, 1), clean_specs_padded.permute(0, 2, 1)

class MelDataset(Dataset):
    def __init__(self, data_dir, data_files, segment_size, sampling_rate, 
                 n_fft, hop_length, win_length, n_mels, f_min, f_max, power,
                 split=True, shuffle=True, **kwargs):
        self.data_dir = data_dir
        self.audio_files = data_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        
        if shuffle:
            random.shuffle(self.audio_files)

        # Spectrogram transform
        self.spectrogram_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power
        )
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        try:
            clean_path_rel, noisy_path_rel = self.audio_files[index]
            clean_path = os.path.join(self.data_dir, clean_path_rel)
            noisy_path = os.path.join(self.data_dir, noisy_path_rel)
            
            clean_audio, _ = load_wav(clean_path, self.sampling_rate)
            noisy_audio, _ = load_wav(noisy_path, self.sampling_rate)

            if self.split:
                if noisy_audio.size(1) >= self.segment_size:
                    max_start = noisy_audio.size(1) - self.segment_size
                    start = random.randint(0, max_start)
                    noisy_audio = noisy_audio[:, start : start + self.segment_size]
                    clean_audio = clean_audio[:, start : start + self.segment_size]
                else:
                    pad_size = self.segment_size - noisy_audio.size(1)
                    noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_size))
                    clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_size))
            
            noisy_spec = self.spectrogram_transform(noisy_audio.squeeze(0))
            clean_spec = self.spectrogram_transform(clean_audio.squeeze(0))

            return noisy_spec, clean_spec
        except Exception as e:
            print(f"WARNING: Error loading file at index {index} ({noisy_path_rel}): {e}. Skipping.")
            return None