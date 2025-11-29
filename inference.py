#!/usr/bin/env python
import torch
import torchaudio
import argparse
import os
from glob import glob
from tqdm import tqdm
from argparse import Namespace
import yaml

# Assumindo que os módulos estão em lightning_modules
from lightning_modules.cleanspecnet_module import CleanSpecNetLightningModule

def run_inference(config: dict):
    """
    Executa a inferência a partir de um dicionário de configuração.
    """
    inf_cfg = config['inference_params']
    model_hparams = config['model_hparams']
    audio_cfg = config['audio_params']
    
    device = torch.device('cuda' if torch.cuda.is_available() and not inf_cfg['cpu'] else 'cpu')
    print(f"Usando dispositivo: {device}")

    print(f"Carregando modelo do checkpoint: {inf_cfg['checkpoint_path']}")
    try:
        # Usa load_from_checkpoint que é o método padrão do Lightning
        model = CleanSpecNetLightningModule.load_from_checkpoint(
            inf_cfg['checkpoint_path'],
            map_location=device,
            **model_hparams # Passa os hparams para recriar o modelo
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    os.makedirs(inf_cfg['output_dir'], exist_ok=True)
    audio_files = glob(os.path.join(inf_cfg['input_dir'], '*.wav'))

    print(f"Processando {len(audio_files)} arquivos de áudio...")
    target_sr = audio_cfg['target_sample_rate']
    n_fft = audio_cfg['n_fft']
    hop_length = audio_cfg['hop_length']

    for file_path in tqdm(audio_files, desc="Processando áudios"):
        try:
            waveform, original_sr = torchaudio.load(file_path)
            waveform = waveform.to(device)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(original_sr, target_sr).to(device)
                waveform = resampler(waveform)

            # Pré-processamento: Extrair espectrograma
            stft = torch.stft(waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True)
            spectrogram = torch.abs(stft).unsqueeze(0)

            with torch.no_grad():
                # O modelo retorna o espectrograma limpo
                enhanced_spectrogram = model(spectrogram)

            # Pós-processamento: Reconstruir áudio com a fase original
            phase = torch.angle(stft)
            enhanced_complex_spec = enhanced_spectrogram.squeeze(0) * torch.exp(1j * phase)
            enhanced_waveform = torch.istft(enhanced_complex_spec, n_fft=n_fft, hop_length=hop_length, length=waveform.shape[-1]).cpu()

            if original_sr != target_sr:
                resampler_back = torchaudio.transforms.Resample(target_sr, original_sr)
                enhanced_waveform = resampler_back(enhanced_waveform)

            output_filename = os.path.join(inf_cfg['output_dir'], os.path.basename(file_path))
            torchaudio.save(output_filename, enhanced_waveform.unsqueeze(0), original_sr)

        except Exception as e:
            print(f"\nErro ao processar o arquivo {file_path}: {e}")

    print(f"Processo de inferência concluído! Resultados salvos em '{inf_cfg['output_dir']}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_inference(config)