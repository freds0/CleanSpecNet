import os
import argparse
import json
from glob import glob
import torchaudio
import soundfile as sf
import torch
import random

from torch_audiomentations import (
    Compose, Gain, AddBackgroundNoise, ApplyImpulseResponse, LowPassFilter, HighPassFilter, BandPassFilter, AddColoredNoise
)

class AudioAugmenter:
    def __init__(self, augmentations, device='cpu', seed=42):
        self.augmentations = augmentations
        self.device = device
        self.seed = seed
        self.compose = self._create_compose(self.augmentations)

    def _create_compose(self, augmentations):
        aug_list = []
        for aug in augmentations:
            name = aug['name']
            params = aug.get('params', {})
            if name == 'AddBackgroundNoise':
                if 'background_paths' not in params:
                    raise ValueError("The 'background_paths' parameter is required for AddBackgroundNoise")
                aug_list.append(AddBackgroundNoise(**params))
            elif name == 'ApplyImpulseResponse':
                if 'ir_paths' not in params:
                    raise ValueError("The 'ir_paths' parameter is required for ApplyImpulseResponse")
                aug_list.append(ApplyImpulseResponse(**params))
            elif name == 'Gain':
                aug_list.append(Gain(**params))                
            elif name == 'LowPassFilter':
                aug_list.append(LowPassFilter(**params))
            elif name == 'HighPassFilter':
                aug_list.append(HighPassFilter(**params))
            elif name == 'BandPassFilter':
                aug_list.append(BandPassFilter(**params))
            elif name == 'AddColoredNoise':
                aug_list.append(AddColoredNoise(**params))
            else:
                print(f"Warning: Unknown augmentation '{name}'")
        return Compose(aug_list, shuffle=True, p=1.0) 

    def apply(self, waveform, sr):
        # Set seed for reproducibility
        if self.seed is not None:
            self.seed += random.randint(1, 1000)
            torch.manual_seed(self.seed) 
            random.seed(self.seed) 
            self.seed += 1            

        # Ensure waveform is a tensor with shape (batch_size, num_channels, num_samples)
        waveform = waveform.reshape(1, 1, -1)
        # Apply augmentations
        augmented_waveform = self.compose(waveform, sample_rate=sr)
        augmented_waveform = augmented_waveform.reshape(1, -1)
        return augmented_waveform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="configs/config.json", help='Path to config.json')
    parser.add_argument('--input_dir', '-i', type=str, default="dataset/wavs", help='Input directory')
    parser.add_argument('--output_dir', '-o', type=str, default="output_wavs", help='Output directory')
    parser.add_argument('--search_pattern', '-s', type=str, default="*.wav", help='Search pattern for input files')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    # Load configurations from config.json
    with open(args.config) as f:
        config = json.load(f)

    from tqdm import tqdm

    # Get augmentations and sample rate from config
    augmentations = config["trainset_config"].get('augmentations', [])
    sample_rate = config.get('sample_rate', 16000)  # Default sample rate if not specified

    print("Augmentations:", augmentations)
    print("Sample rate:", sample_rate)
    print("Device:", args.device)
    print("Seed:", args.seed)

    augmenter = AudioAugmenter(augmentations, device=args.device, seed=args.seed)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process each file
    for file in tqdm(glob(os.path.join(args.input_dir, args.search_pattern))):
        print(f"Processing {file}")
        # Load audio file using torchaudio
        waveform, sr = torchaudio.load(file)
        if sr != sample_rate:
            # Resample if necessary
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        augmented_waveform = augmenter.apply(waveform, sr)
        output_file = os.path.join(args.output_dir, os.path.basename(file))

        # Transpose to (num_samples, num_channels) for saving
        augmented_waveform = augmented_waveform.T

        sf.write(output_file, augmented_waveform, sr)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()