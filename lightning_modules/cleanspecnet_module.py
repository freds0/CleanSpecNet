# lightning_modules/cleanspecnet_module.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio.transforms as T

import matplotlib
# Define o backend não interativo. Deve ser feito ANTES de importar pyplot.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import io
import numpy as np
from PIL import Image

from cleanspecnet.cleanspecnet import CleanSpecNet

# --- FUNÇÃO AUXILIAR PARA PLOTAR ESPECTROGRAMAS COLORIDOS ---
def plot_spectrogram_to_tensor(spectrogram):
    """
    Converte um tensor de espectrograma 2D em um tensor de imagem RGB
    usando um mapa de cores para visualização no TensorBoard.
    """
    # Normaliza o espectrograma para o range [0, 1] para melhor visualização
    # Isso evita que imagens fiquem muito escuras ou claras demais
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if spec_max > spec_min:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    # Usa o mapa de cores 'viridis', comum para espectrogramas
    im = ax.imshow(spectrogram, cmap='viridis', aspect='auto', origin='lower')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Salva a figura em um buffer de memória em vez de um arquivo
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # Lê a imagem do buffer e converte para o formato RGB
    image = Image.open(buf).convert('RGB')
    image_np = np.array(image)
    
    # Converte de (Altura, Largura, Canais) para (Canais, Altura, Largura) para o TensorBoard
    return torch.from_numpy(image_np).permute(2, 0, 1)

class CleanSpecNetLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = CleanSpecNet(
            input_channels=self.hparams.input_channels,
            num_conv_layers=self.hparams.num_conv_layers,
            kernel_size=self.hparams.kernel_size,
            stride=self.hparams.stride,
            hidden_dim=self.hparams.hidden_dim,
            num_attention_layers=self.hparams.num_attention_layers,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout
        )
        
        self.criterion = torch.nn.L1Loss()
        
        # Cria a instância do GriffinLim usando os hparams
        self.griffin_lim = T.GriffinLim(
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            power=1.0 # Assumindo espectrograma de magnitude
        )

    def forward(self, spectrogram):
        return self.model(spectrogram)

    def training_step(self, batch, batch_idx):
        noisy_spec, clean_spec = batch
        enhanced_spec = self(noisy_spec)
        loss = self.criterion(enhanced_spec, clean_spec)
        
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_spec, clean_spec = batch
        enhanced_spec = self(noisy_spec)
        loss = self.criterion(enhanced_spec, clean_spec)
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            tensorboard = self.logger.experiment
            
            # Pega a primeira amostra e move para a CPU para plotagem
            noisy_sample_spec = noisy_spec[0].cpu()
            clean_sample_spec = clean_spec[0].cpu()
            enhanced_sample_spec = enhanced_spec[0].cpu()

            # --- Log das Imagens Coloridas ---
            tensorboard.add_image("Val_Spectrogram/Input", plot_spectrogram_to_tensor(noisy_sample_spec), self.current_epoch)
            tensorboard.add_image("Val_Spectrogram/Output", plot_spectrogram_to_tensor(enhanced_sample_spec), self.current_epoch)
            tensorboard.add_image("Val_Spectrogram/Target", plot_spectrogram_to_tensor(clean_sample_spec), self.current_epoch)

            # --- Log dos Áudios ---
            self.griffin_lim.to(self.device)
            noisy_audio = self.griffin_lim(noisy_sample_spec.to(self.device))
            enhanced_audio = self.griffin_lim(enhanced_sample_spec.to(self.device))
            clean_audio = self.griffin_lim(clean_sample_spec.to(self.device))

            tensorboard.add_audio("Val_Audio/Input", noisy_audio.unsqueeze(0), self.current_epoch, self.hparams.sample_rate)
            tensorboard.add_audio("Val_Audio/Output", enhanced_audio.unsqueeze(0), self.current_epoch, self.hparams.sample_rate)
            tensorboard.add_audio("Val_Audio/Target", clean_audio.unsqueeze(0), self.current_epoch, self.hparams.sample_rate)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
        #return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return {
            "optimizer": optimizer
        }
