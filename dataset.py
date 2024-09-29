import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import random
from audiomentations import Compose, Mp3Compression, AddGaussianSNR, AddBackgroundNoise, PolarityInversion, LowPassFilter, HighPassFilter
from augmentation import AudioAugmenter

class PairSpecDataset(Dataset):
    """
    Create a Dataset of spectrogram.
    Each element is a tuple of the form (clean spectrogram, noisy spectrogram
    """

    def __init__(self, data_dir='./dataset', subset='train', train_metadata='train.csv', 
                 test_metadata='test.csv', audio_min_sec=1.0, crop_length_sec=0, sample_rate=24000, n_fft=1024, 
                 hop_length=256, win_length=1024, power=1.0, augmentations=None):
        
        super(PairSpecDataset, self).__init__()
        self.sample_rate = sample_rate
        self.spectrogram_fn = T.Spectrogram(n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, power=power, normalized=True, center=False)
        '''
        self.mel_spectrogram_fn = T.MelSpectrogram(
            sample_rate=sample_rate,          # Sample rate of the audio file
            n_fft=1024,                       # FFT window size (typically 1024 or 2048)
            hop_length=hop_length,            # Hop length (distance between successive windows)
            n_mels=80,                        # Number of Mel bands
        )
        '''
        assert subset in ["train", "test"], "Subset must be 'train' or 'test'"
        self.audio_min_sec = audio_min_sec
        self.crop_length_sec = crop_length_sec
        self.subset = subset

        # Read the metadata file to get file paths and labels
        if subset == "train":
            with open(train_metadata, 'r') as f:
                data = f.readlines()
        else:
            with open(test_metadata, 'r') as f:
                data = f.readlines()  

        # Extract file names and corresponding labels
        self.files = [os.path.join(data_dir, line.strip().split("|")[0]) for line in data]

        # Initialize the AudioAugmenter with the provided augmentations, if any
        self.audio_augmenter = AudioAugmenter(augmentations) if augmentations else None
        if self.audio_augmenter:
            print("Using audio augmentations")

    @staticmethod
    def _load_audio_and_resample(filepath : str, target_sr : int = 24000) -> torch.Tensor:
        """
        Loads audio and resamples to target sampling rate. Returns single channel.

        Args:
            audio_path (str): Path to audio file.
            target_sr (int, optional): Target sampling rate. Defaults to 16000.

        Returns:
            torch.Tensor: Tensor of audio file, returns single channel only.
        """
        try:
            audio_wav, sr = torchaudio.load(filepath, normalize=True)
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")            
            return False
                    
        # Resample the audio if necessary
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            audio_wav = resampler(audio_wav)

        # Convert multi-channel audio to mono by averaging channels
        if audio_wav.dim() > 1 and audio_wav.shape[0] > 1:
            audio_wav = torch.mean(audio_wav, dim=0)
        else:
            audio_wav = audio_wav.squeeze(0)
            
        return audio_wav
    

    def __getitem__(self, index):

        # Get the file path and label for the index-th item
        filepath = self.files[index]
        
        # Load the audio file
        audio = self._load_audio_and_resample(filepath, self.sample_rate)
        audio_len_sec = len(audio) / self.sample_rate
        if type(audio) == bool and audio == False or (audio_len_sec < self.audio_min_sec):
            # If there's an error, try loading the next audio file
            return self.__getitem__((index + 1) % len(self))

        # Normalize the audio to have values between -1 and 1
        #audio = audio / audio.abs().max()

        # Ensure audio length is a multiple of hop_length
        num_frames = int(np.ceil((len(audio) - self.spectrogram_fn.n_fft) / self.spectrogram_fn.hop_length)) + 1
        total_length = (num_frames - 1) * self.spectrogram_fn.hop_length + self.spectrogram_fn.n_fft
        padding = total_length - len(audio)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        # Apply cropping if specified
        crop_length = int(self.crop_length_sec * self.sample_rate)
        if crop_length > 0 and crop_length < len(audio):
            # Randomly select a starting point for cropping
            start = np.random.randint(low=0, high=len(audio) - crop_length + 1)
            audio = audio[start:(start + crop_length)]

        # Apply audio augmentations if any
        if self.audio_augmenter:
            # Convert the tensor to a NumPy array for augmentation
            audio_np = audio.numpy()
            # Apply the augmentations
            audio_np = self.audio_augmenter.apply(audio_np, self.sample_rate)
            # Convert back to a tensor
            noisy_audio = torch.from_numpy(audio_np)
        else:
            # If there are no augmentations, noisy audio is the same as clean audio.
            noisy_audio = audio.clone()

        clean_audio = audio

        # Ensure both audios are the same size
        min_length = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_length]
        noisy_audio = noisy_audio[:min_length]

        # Add channel dimension
        clean_audio = clean_audio.unsqueeze(0)
        noisy_audio = noisy_audio.unsqueeze(0)

        # Get the spectrograms
        clean_spec = self.spectrogram_fn(clean_audio).squeeze()
        noisy_spec = self.spectrogram_fn(noisy_audio).squeeze()
        #clean_spec = self.mel_spectrogram_fn(clean_audio).squeeze()
        #noisy_spec = self.mel_spectrogram_fn(noisy_audio).squeeze()

        '''
        original_length_in_samples = clean_audio.shape[-1]        
        original_length_in_seconds = original_length_in_samples / self.sample_rate
        length_in_samples = (clean_spec.shape[1] - 1) * self.spectrogram_fn.hop_length + self.spectrogram_fn.n_fft
        length_in_seconds = length_in_samples / self.sample_rate
        print(filepath)
        print(f"Original length: {original_length_in_seconds:.2f} s ({original_length_in_samples} samples)")
        print(f"Processed length: {length_in_seconds:.2f} s ({length_in_samples} samples)")

        print(f"Clean spec shape:", clean_spec.shape)
        '''
        return (noisy_spec, clean_spec)


    def __len__(self):
        return len(self.files)


def collate_fn(batch):
    noisy_specs, clean_specs = zip(*batch)
    
    max_time = max([spec.shape[1] for spec in clean_specs])

    padded_clean_specs = []
    padded_noisy_specs = []
    # Fill the spectrograms to be the same size
    for clean_spec, noisy_spec in zip(clean_specs, noisy_specs):
        pad_time = max_time - clean_spec.shape[1]
        # There is no padding on the frequency, so pad_top and pad_bottom are zero
        pad = (0, pad_time, 0, 0)  # (pad_left, pad_right, pad_top, pad_bottom)
        padded_clean_spec = F.pad(clean_spec, pad) 
        padded_noisy_spec = F.pad(noisy_spec, pad)
        padded_clean_specs.append(padded_clean_spec)
        padded_noisy_specs.append(padded_noisy_spec)
    
    # Stack the tensors to form the batch
    clean_batch = torch.stack(padded_clean_specs)
    noisy_batch = torch.stack(padded_noisy_specs)

    return noisy_batch, clean_batch


def load_PairSpecDataset(data_dir, train_metadata, test_metadata, audio_min_sec, crop_length_sec, batch_size, sample_rate,
                               n_fft, hop_length, win_length, power, augmentations=None, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    train_dataset = PairSpecDataset(data_dir=data_dir, subset='train', train_metadata=train_metadata, test_metadata=test_metadata, 
                                    audio_min_sec=audio_min_sec, crop_length_sec=crop_length_sec, sample_rate=sample_rate, 
                                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=power, 
                                    augmentations=augmentations)    
    test_dataset = PairSpecDataset(data_dir=data_dir, subset='test', train_metadata=train_metadata, test_metadata=test_metadata,
                                    audio_min_sec=audio_min_sec, crop_length_sec=crop_length_sec, sample_rate=sample_rate, 
                                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=power, 
                                    augmentations=augmentations)
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, shuffle=True, **kwargs)
        
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    import json
    with open('./configs/config.json') as f:
        config = json.load(f)
    trainset_config = config["trainset_config"]
    augmentations = config.get("augmentations", None)

    trainloader, testloader = load_PairSpecDataset(**trainset_config, batch_size=2, augmentations=augmentations, num_gpus=1)
    print("Data loaded")

    print(len(trainloader), len(testloader))

    for i, (noisy_spec, clean_spec) in enumerate(trainloader): 
        print(clean_spec.shape)
        print(clean_spec.max(), clean_spec.min())        
        
        print(noisy_spec.shape)
        print(noisy_spec.max(), noisy_spec.min())
        if i > 3:
            break
