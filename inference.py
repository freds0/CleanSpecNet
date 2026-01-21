#!/usr/bin/env python
"""
CleanSpecNet Inference Script

This script performs audio denoising using a trained CleanSpecNet model.
It processes audio files from an input directory and saves enhanced
versions to an output directory.
"""

import torch
import torchaudio
import argparse
import os
from glob import glob
from tqdm import tqdm
from argparse import Namespace
import yaml

from lightning_modules.cleanspecnet_module import CleanSpecNetLightningModule


def run_inference(config: dict):
    """
    Run inference using a configuration dictionary.

    Args:
        config: Dictionary containing inference_params, model_hparams, and audio_params
    """
    inf_cfg = config['inference_params']
    model_hparams = config['model_hparams']
    audio_cfg = config['audio_params']

    device = torch.device('cuda' if torch.cuda.is_available() and not inf_cfg['cpu'] else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from checkpoint: {inf_cfg['checkpoint_path']}")
    try:
        # Use load_from_checkpoint which is the standard Lightning method
        model = CleanSpecNetLightningModule.load_from_checkpoint(
            inf_cfg['checkpoint_path'],
            map_location=device,
            **model_hparams  # Pass hparams to recreate the model
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    os.makedirs(inf_cfg['output_dir'], exist_ok=True)
    audio_files = glob(os.path.join(inf_cfg['input_dir'], '*.wav'))

    print(f"Processing {len(audio_files)} audio files...")
    target_sr = audio_cfg['target_sample_rate']
    n_fft = audio_cfg['n_fft']
    hop_length = audio_cfg['hop_length']

    for file_path in tqdm(audio_files, desc="Processing audio"):
        try:
            waveform, original_sr = torchaudio.load(file_path)
            waveform = waveform.to(device)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(original_sr, target_sr).to(device)
                waveform = resampler(waveform)

            # Preprocessing: Extract spectrogram
            stft = torch.stft(waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True)
            spectrogram = torch.abs(stft).unsqueeze(0)

            with torch.no_grad():
                # Model returns the clean spectrogram
                enhanced_spectrogram = model(spectrogram)

            # Postprocessing: Reconstruct audio using original phase
            phase = torch.angle(stft)
            enhanced_complex_spec = enhanced_spectrogram.squeeze(0) * torch.exp(1j * phase)
            enhanced_waveform = torch.istft(enhanced_complex_spec, n_fft=n_fft, hop_length=hop_length, length=waveform.shape[-1]).cpu()

            # Resample back to original sample rate if needed
            if original_sr != target_sr:
                resampler_back = torchaudio.transforms.Resample(target_sr, original_sr)
                enhanced_waveform = resampler_back(enhanced_waveform)

            output_filename = os.path.join(inf_cfg['output_dir'], os.path.basename(file_path))
            torchaudio.save(output_filename, enhanced_waveform.unsqueeze(0), original_sr)

        except Exception as e:
            print(f"\nError processing file {file_path}: {e}")

    print(f"Inference complete! Results saved to '{inf_cfg['output_dir']}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CleanSpecNet inference on audio files.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_inference(config)