import os
import random
import torch
import torch.utils.data
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensures multiprocessing works correctly
if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)

MAX_WAV_VALUE = 32768.0

def load_wav(full_path, target_sr):
    """Loads a .wav file and resamples it to the target sampling rate."""
    data, sampling_rate = torchaudio.load(full_path, normalize=True)
    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        data = resampler(data)
    return data

def get_dataset_filelist(filelist_path):
    """Reads the filelist and returns pairs of paths (clean, noisy)."""
    with open(filelist_path, 'r', encoding='utf-8') as f:
        files = [line.strip().split('|') for line in f]
    return files

class SpectrogramSavingDataset(torch.utils.data.Dataset):
    """
    A Dataset class that loads audio files, generates their spectrograms,
    and saves them to disk.
    """
    def __init__(self, data_dir, data_files, output_dir, segment_size=8192,
                 n_fft=1024, hop_size=256, win_size=1024, sampling_rate=16000, 
                 shuffle=True):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.audio_files = get_dataset_filelist(data_files)

        if shuffle:
            random.seed(1234)
            random.shuffle(self.audio_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size

        # Torchaudio spectrogram transform function
        self.spectrogram_fn = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            power=1.0,        # For magnitude, not power (power=2.0)
            normalized=False, # Normalization is done manually later if needed
            center=False
        )

    def __getitem__(self, index):
        # Get relative paths from the filelist
        clean_relative_path, noisy_relative_path = self.audio_files[index]

        # Build full input paths
        clean_filepath = os.path.join(self.data_dir, clean_relative_path)
        noisy_filepath = os.path.join(self.data_dir, noisy_relative_path)

        try:
            # Load audio files
            clean_audio = load_wav(clean_filepath, self.sampling_rate)
            noisy_audio = load_wav(noisy_filepath, self.sampling_rate)

            # Ensure both audios have the same length
            min_len = min(clean_audio.size(1), noisy_audio.size(1))
            clean_audio = clean_audio[:, :min_len]
            noisy_audio = noisy_audio[:, :min_len]

            # Segment audio into fixed-size chunks
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)

                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start:audio_start + self.segment_size]
            else:
                # Pad if audio is shorter than segment size
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

            # Generate spectrograms
            clean_spec = self.spectrogram_fn(clean_audio).squeeze(0)  # Remove channel dimension
            input_spec = self.spectrogram_fn(noisy_audio).squeeze(0)  # Remove channel dimension

            # ======================================================
            # Spectrogram Saving Logic
            # ======================================================

            # Remove original extension (.wav) to add new one (.pt)
            base_clean_path, _ = os.path.splitext(clean_filepath)
            base_noisy_path, _ = os.path.splitext(noisy_filepath)

            # Build output path, preserving subdirectory structure
            clean_spec_path = base_clean_path + '.pt'
            noisy_spec_path = base_noisy_path + '.pt'

            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(clean_spec_path), exist_ok=True)
            os.makedirs(os.path.dirname(noisy_spec_path), exist_ok=True)

            # Save spectrogram tensors
            torch.save(clean_spec, clean_spec_path)
            torch.save(input_spec, noisy_spec_path)

            # Return saved file paths for confirmation
            return clean_spec_path, noisy_spec_path

        except Exception as e:
            print(f"Error processing file {clean_filepath}: {e}")
            return None, None  # Return None on error

    def __len__(self):
        return len(self.audio_files)

def collate_fn_filter_none(batch):
    """Collate function that filters out failed samples (None)."""
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else ([], [])


# ======================================================
# Main Execution Block
# ======================================================
if __name__ == '__main__':

    # --- Configuration ---
    # Change these paths to match your file structure
    DATA_DIR = '/root/DATASETS/VoiceBank-DEMAND-16k'       # Root directory where audio files are located
    FILE_LIST_PATH = 'filelists/train.csv'                 # File with "clean|noisy" pairs
    OUTPUT_DIR = '/root/DATASETS/VoiceBank-DEMAND-16k/'    # Where spectrograms will be saved

    # Dataset parameters
    SAMPLING_RATE = 16000
    SEGMENT_SIZE = SAMPLING_RATE * 4  # 4-second segments
    N_FFT = 1024
    HOP_SIZE = 256
    WIN_SIZE = 1024
    BATCH_SIZE = 16   # Number of files to process in parallel
    NUM_WORKERS = 4   # Number of processes to speed up preprocessing

    # --- Execution ---
    print("Starting spectrogram preprocessing and saving...")

    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instantiate the Dataset
    dataset = SpectrogramSavingDataset(
        data_dir=DATA_DIR,
        data_files=FILE_LIST_PATH,
        output_dir=OUTPUT_DIR,
        sampling_rate=SAMPLING_RATE,
        segment_size=SEGMENT_SIZE,
        n_fft=N_FFT,
        hop_size=HOP_SIZE,
        win_size=WIN_SIZE,
        shuffle=False  # Shuffle is not needed for saving only
    )

    # Create DataLoader for parallel processing
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn_filter_none  # Use custom collate function
    )

    # Iterate over the dataset with a progress bar (tqdm)
    for clean_paths, noisy_paths in tqdm(data_loader, desc="Saving spectrograms"):
        # The saving logic is already inside the dataset's __getitem__.
        # The loop just forces iteration over all data.
        pass

    print("\nProcessing complete!")
    print(f"All spectrograms saved to: {os.path.abspath(OUTPUT_DIR)}")
