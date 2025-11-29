import os
import random
import torch
import torch.utils.data
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm # Ótimo para barras de progresso!

# Garante que o multiprocessing funcione corretamente
if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)

MAX_WAV_VALUE = 32768.0

def load_wav(full_path, target_sr):
    """Carrega um arquivo .wav e o reamostra para a taxa de amostragem alvo."""
    data, sampling_rate = torchaudio.load(full_path, normalize=True)
    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        data = resampler(data)
    return data

def get_dataset_filelist(filelist_path):
    """Lê o arquivo de lista e retorna pares de caminhos (limpo, com ruído)."""
    with open(filelist_path, 'r', encoding='utf-8') as f:
        files = [line.strip().split('|') for line in f]
    return files

class SpectrogramSavingDataset(torch.utils.data.Dataset):
    """
    Uma classe de Dataset que carrega áudios, gera seus espectrogramas
    e os salva em disco.
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

        # A função de transformação de espectrograma do torchaudio
        self.spectrogram_fn = T.Spectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_size, 
            win_length=self.win_size, 
            power=1.0,        # Para magnitude, não potência (power=2.0)
            normalized=False, # Normalização é feita manualmente depois se necessário
            center=False
        )

    def __getitem__(self, index):
        # Pega os caminhos relativos do arquivo de lista
        clean_relative_path, noisy_relative_path = self.audio_files[index]

        # Monta os caminhos completos de entrada
        clean_filepath = os.path.join(self.data_dir, clean_relative_path)
        noisy_filepath = os.path.join(self.data_dir, noisy_relative_path)

        try:
            # Carrega os áudios
            clean_audio = load_wav(clean_filepath, self.sampling_rate)
            noisy_audio = load_wav(noisy_filepath, self.sampling_rate)

            # Garante que ambos os áudios tenham o mesmo comprimento
            min_len = min(clean_audio.size(1), noisy_audio.size(1))
            clean_audio = clean_audio[:, :min_len]
            noisy_audio = noisy_audio[:, :min_len]
            
            # Segmenta o áudio em pedaços de tamanho fixo
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                
                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start:audio_start + self.segment_size]
            else:
                # Faz padding se o áudio for menor que o segmento
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

            # Gera os espectrogramas
            clean_spec = self.spectrogram_fn(clean_audio).squeeze(0) # Remove a dimensão do canal
            input_spec = self.spectrogram_fn(noisy_audio).squeeze(0) # Remove a dimensão do canal
            
            # ======================================================
            # Início da Lógica para Salvar os Espectrogramas
            # ======================================================

            # Remove a extensão original (.wav) para adicionar a nova (.pt)
            base_clean_path, _ = os.path.splitext(clean_filepath)
            base_noisy_path, _ = os.path.splitext(noisy_filepath)

            # Constrói o caminho de saída, preservando a estrutura de subdiretórios
            #clean_spec_path = os.path.join(self.output_dir, 'clean', base_clean_path + '.pt')
            #noisy_spec_path = os.path.join(self.output_dir, 'noisy', base_noisy_path + '.pt')
            clean_spec_path = base_clean_path + '.pt'
            noisy_spec_path = base_noisy_path + '.pt'

            # Cria os diretórios de saída se eles não existirem
            os.makedirs(os.path.dirname(clean_spec_path), exist_ok=True)
            os.makedirs(os.path.dirname(noisy_spec_path), exist_ok=True)

            # Salva os tensores de espectrograma
            torch.save(clean_spec, clean_spec_path)
            torch.save(input_spec, noisy_spec_path)
            
            # Retorna os caminhos dos arquivos salvos para confirmação
            return clean_spec_path, noisy_spec_path

        except Exception as e:
            print(f"Erro ao processar o arquivo {clean_filepath}: {e}")
            return None, None # Retorna None em caso de erro

    def __len__(self):
        return len(self.audio_files)

def collate_fn_filter_none(batch):
    """Função de collate que filtra amostras que falharam (None)."""
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else ([], [])


# ======================================================
# Bloco Principal de Execução
# ======================================================
if __name__ == '__main__':
    
    # --- Configurações ---
    # ⚠️ Altere estes caminhos para corresponder à sua estrutura de arquivos
    DATA_DIR = '/root/DATASETS/VoiceBank-DEMAND-16k'       # Diretório raiz onde os áudios estão
    FILE_LIST_PATH = 'filelists/train.csv' # Arquivo com os pares "clean|noisy"
    OUTPUT_DIR = '/root/DATASETS/VoiceBank-DEMAND-16k/'  # Onde os espectrogramas serão salvos
    
    # Parâmetros do Dataset
    SAMPLING_RATE = 16000
    SEGMENT_SIZE = SAMPLING_RATE * 4  # Segmentos de 4 segundos
    N_FFT = 1024
    HOP_SIZE = 256
    WIN_SIZE = 1024
    BATCH_SIZE = 16 # Número de arquivos a processar em paralelo
    NUM_WORKERS = 4 # Número de processos para acelerar o pré-processamento

    # --- Execução ---
    print("🚀 Iniciando o pré-processamento e salvamento dos espectrogramas...")

    # Cria o diretório de saída principal
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instancia o Dataset
    dataset = SpectrogramSavingDataset(
        data_dir=DATA_DIR,
        data_files=FILE_LIST_PATH,
        output_dir=OUTPUT_DIR,
        sampling_rate=SAMPLING_RATE,
        segment_size=SEGMENT_SIZE,
        n_fft=N_FFT,
        hop_size=HOP_SIZE,
        win_size=WIN_SIZE,
        shuffle=False # Shuffle não é necessário para apenas salvar
    )

    # Cria o DataLoader para processar em paralelo
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn_filter_none # Usa o collate customizado
    )

    # Itera sobre o dataset com uma barra de progresso (tqdm)
    for clean_paths, noisy_paths in tqdm(data_loader, desc="Salvando espectrogramas"):
        # A lógica de salvamento já está dentro do __getitem__ do dataset.
        # O loop apenas força a iteração sobre todos os dados.
        pass

    print("\n✅ Processamento concluído!")
    print(f"Todos os espectrogramas foram salvos em: {os.path.abspath(OUTPUT_DIR)}")
