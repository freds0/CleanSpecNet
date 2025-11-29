# lightning_modules/data_module.py (Corrigido)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from spec_dataset import MelDataset, get_dataset_filelist, custom_collate_fn

class CleanSpecNetDataModule(pl.LightningDataModule):
    # __init__ agora aceita todos os parâmetros do seu config.yaml
    def __init__(self, data_dir: str, train_list_path: str, val_list_path: str, test_list_path: str, 
                 batch_size: int, num_workers: int, segment_size: int, n_fft: int, n_mels: int, 
                 hop_length: int, win_length: int, sampling_rate: int, f_min: int, f_max: int, power: float, **kwargs):
        super().__init__()
        # Salva todos os parâmetros em self.hparams para fácil acesso
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        # Agrupa todos os parâmetros necessários para o MelDataset
        # Esta lista agora está completa
        self.dataset_kwargs = {
            "segment_size": self.hparams.segment_size,
            "n_fft": self.hparams.n_fft,
            "n_mels": self.hparams.n_mels,
            "hop_length": self.hparams.hop_length, # 👈 Adicionado
            "win_length": self.hparams.win_length, # 👈 Adicionado
            "sampling_rate": self.hparams.sampling_rate,
            "f_min": self.hparams.f_min,
            "f_max": self.hparams.f_max,
            "power": self.hparams.power             # 👈 Adicionado
        }

        if stage == "fit" or stage is None:
            train_files = get_dataset_filelist(self.hparams.train_list_path)
            val_files = get_dataset_filelist(self.hparams.val_list_path)
            
            self.train_dataset = MelDataset(
                data_dir=self.hparams.data_dir,
                data_files=train_files,
                split=True,
                **self.dataset_kwargs
            )
            self.val_dataset = MelDataset(
                data_dir=self.hparams.data_dir,
                data_files=val_files,
                split=False,
                shuffle=False,
                **self.dataset_kwargs
            )

        if stage == "test" or stage is None:
            test_files = get_dataset_filelist(self.hparams.test_list_path)
            self.test_dataset = MelDataset(
                data_dir=self.hparams.data_dir,
                data_files=test_files,
                split=False,
                shuffle=False,
                **self.dataset_kwargs
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, collate_fn=custom_collate_fn, pin_memory=True)