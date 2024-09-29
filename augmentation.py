import os
import argparse
import json
from glob import glob
import librosa
import soundfile as sf

from audiomentations import (
    Compose, Mp3Compression, AddGaussianSNR, AddBackgroundNoise,
    PolarityInversion, LowPassFilter, HighPassFilter, ApplyImpulseResponse
)

class AudioAugmenter:
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.compose = self._create_compose(self.augmentations)

    def _create_compose(self, augmentations):
        aug_list = []
        for aug in augmentations:
            name = aug['name']
            params = aug.get('params', {})
            if name == 'Mp3Compression':
                aug_list.append(Mp3Compression(**params))
            elif name == 'AddGaussianSNR':
                aug_list.append(AddGaussianSNR(**params))
            elif name == 'AddBackgroundNoise':
                aug_list.append(AddBackgroundNoise(**params))
            elif name == 'LowPassFilter':
                aug_list.append(LowPassFilter(**params))
            elif name == 'HighPassFilter':
                aug_list.append(HighPassFilter(**params))
            elif name == 'ApplyImpulseResponse':
                aug_list.append(ApplyImpulseResponse(**params))                
            else:
                print(f"Warning: Unknown augmentation '{name}'")
        return Compose(aug_list)

    def apply(self, waveform, sr):
        augmented_waveform = self.compose(samples=waveform, sample_rate=sr)
        return augmented_waveform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="configs/config.json", help='Path to config.json')
    parser.add_argument('--input_dir', '-i', type=str, default="dataset/wavs", help='Input directory')
    parser.add_argument('--output_dir', '-o', type=str, default="output_wavs", help='Output directory')
    parser.add_argument('--search_pattern', '-s', type=str, default="*.wav", help='Search pattern for input files')
    args = parser.parse_args()

    # Load configurations from config.json
    with open(args.config) as f:
        config = json.load(f)

    # Get augmentations and sample rate from config
    augmentations = config["trainset_config"].get('augmentations', [])
    sample_rate = config.get('sample_rate', 16000)  # Default sample rate if not specified

    print("Augmentations:", augmentations)
    print("Sample rate:", sample_rate)

    augmenter = AudioAugmenter(augmentations)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process each file
    for file in glob(os.path.join(args.input_dir, args.search_pattern)):
        print(f"Processing {file}")
        waveform, sr = librosa.load(file, sr=sample_rate)
        augmented_waveform = augmenter.apply(waveform, sr)
        output_file = os.path.join(args.output_dir, os.path.basename(file))

        sf.write(output_file, augmented_waveform, sr)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
