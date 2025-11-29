
import os
import argparse
from glob import glob
from os.path import join, basename, isdir
import torch, torchaudio
from torchaudio.functional import resample
from torch.nn.functional import pad
from tqdm import tqdm
#from torchaudio.pipelines import SQUIM_OBJECTIVE

from pesq import pesq
from pystoi import stoi

class ObjectiveMetricsPredictor:
    def __init__(self, device=None):        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.objective_predictor = SQUIM_OBJECTIVE.get_model().to(self.device)
        #self.objective_predictor.eval()

    def si_snr(self, estimate, reference, epsilon=1e-8):
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
        mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
        scale = mix_pow / (reference_pow + epsilon)

        reference = scale * reference
        error = estimate - reference

        reference_pow = reference.pow(2)
        error_pow = error.pow(2)

        reference_pow = reference_pow.mean(axis=1)
        error_pow = error_pow.mean(axis=1)

        si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
        return si_snr.item()


    def _load_file(self, filepath: str)->(torch.Tensor, int):
        try:
            waveform, sr = torchaudio.load(filepath)
            assert sr >= 16000, "Sample rate must be at least 16kHz"                
            if sr != 16000:
                waveform = resample(waveform, sr, 16000)
                sr = 16000
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return False
        return waveform, sr
    
    def __call__(self, input_str_clean, input_str_distorted)->float:
        if isdir(input_str_clean):
            return self.predict_folder(input_str_clean, input_str_distorted)
        else:
            return self.predict_file(input_str_clean,input_str_distorted)


    def predict_file(self, filepath_clean: str, filepath_distorted: str)->float:
        loaded_result = self._load_file(filepath_clean)
        if not loaded_result:
            print(f"Failed to load file: {filepath_clean}")
            return None
        waveform_clean, sr = loaded_result

        loaded_result = self._load_file(filepath_distorted)
        if not loaded_result:
            print(f"Failed to load file: {filepath_distorted}")
            return None
        waveform_distorted, sr = loaded_result

        pesq_ref = pesq(16000, waveform_clean.numpy(), waveform_distorted.numpy(), mode="wb")
        stoi_ref = stoi(waveform_clean.numpy(), waveform_distorted.numpy(), 16000, extended=False)
        si_sdr_ref = self.si_snr(waveform_distorted.unsqueeze(0), waveform_clean.unsqueeze(0))

        return {
            "stoi": stoi_ref.to('cpu').item(),
            "pesq": pesq_ref.to('cpu').item(),
            "si_sdr": si_sdr_ref.to('cpu').item()
        }          

    def predict_metrics(self, waveform_clean: str, waveform_distorted: str)->float:
        waveform_clean = waveform_clean.squeeze().cpu()
        waveform_distorted = waveform_distorted.squeeze().cpu()        

        assert waveform_clean.ndim == 1, f"Expected 1D waveform_clean, got {waveform_clean.shape}"
        assert waveform_distorted.ndim == 1, f"Expected 1D waveform_distorted, got {waveform_distorted.shape}"

        pesq_ref = pesq(16000, waveform_clean.numpy(), waveform_distorted.numpy(), mode="wb")
        stoi_ref = stoi(waveform_clean.numpy(), waveform_distorted.numpy(), 16000, extended=False)
        si_sdr_ref = self.si_snr(waveform_distorted.unsqueeze(0), waveform_clean.unsqueeze(0))

        return {
            "stoi": stoi_ref,
            "pesq": pesq_ref,
            "si_sdr": si_sdr_ref
        }

    def _collate_fn(self, batch):
        # pad sequences to have same length
        batch = [item.squeeze() for item in batch if item is not None]
        max_length = max([item.shape[0] for item in batch])
        batch = [pad(item, (0, max_length - item.shape[0])) for item in batch]
        return torch.stack(batch, dim=0)
    
    def _predict_batch(self, fileslist_clean, fileslist_distorted):
        loaded_data = self._load_file(fileslist_clean[0])
        if loaded_data is None:
            return None
        if not loaded_data:
            return None
        _, sr = loaded_data

        batch_list_clean = [self._load_file(filepath)[0].squeeze() for filepath in fileslist_clean]
        batch_list_distorted = [self._load_file(filepath)[0].squeeze() for filepath in fileslist_distorted]
        batch_clean = self._collate_fn(batch_list_clean).squeeze()
        batch_distorted = self._collate_fn(batch_list_distorted).squeeze()

        pesq_ref = pesq(16000, batch_clean.numpy(), batch_distorted.numpy(), mode="wb")
        stoi_ref = stoi(batch_clean.numpy(), batch_distorted.numpy(), 16000, extended=False).item()
        si_sdr_ref = si_snr(batch_distorted.unsqueeze(0), batch_clean.unsqueeze(0))
                
        return stoi_ref, pesq_ref, si_sdr_ref
    
    def _create_batchs(self, fileslist, filelist_distorted, batch_size=1):
        for i in range(0, len(fileslist), batch_size):
            yield fileslist[i:i+batch_size], filelist_distorted[i:i+batch_size]

    def predict_folder(self, dirpath_clean: str, dirpath_distorted: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        scores = {}
        filelist_clean = sorted(glob(join(dirpath_clean, search_str)))
        filelist_distorted = sorted(glob(join(dirpath_distorted, search_str)))
        assert len(filelist_clean) == len(filelist_distorted), "Number of files in clean and distorted folders must be the same"

        total_batches = (len(filelist_clean) + batch_size - 1) // batch_size
        for batch_clean, batch_distorted in tqdm(self._create_batchs(filelist_clean, filelist_distorted, batch_size), total=total_batches, desc="Processing Batches"):
            metrics = self._predict_batch(batch_clean, batch_distorted)
            if metrics is None:
                continue
            stoi, pesq, si_sdr = metrics
            filepath = os.path.basename(batch_clean[0])
            '''
            for i, filepath in enumerate(batch_distorted):
                stoi = metrics[0][i]
                pesq = metrics[1][i]
                si_sdr = metrics[2][i] if metrics[2][i] > 0 else 0
                #score = (stoi + (pesq / 4.5) + (si_sdr / 35.0) ) / 3.0
                scores[filepath] = {
                    "stoi": metrics[0][i].item(),
                    "pesq": metrics[1][i].item(),
                    "si_sdr": metrics[2][i].item()#,
                    #"score": score
                }                    
            '''       
            scores[filepath] = {
                "stoi": stoi,
                "pesq": pesq,
                "si_sdr": si_sdr
            }                     
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_clean',      '-i', type=str, default='samples/0', help='Input clean folder')
    parser.add_argument('--input_dir_dist',       '-r', type=str, default='samples/0', help='Input distorted folder')
    parser.add_argument('--output_file',          '-o', type=str, default='metrics_prediction.json', help='Output json filepath')
    parser.add_argument('--batch_size',           '-b', type=int, default=1, help='Batch size for processing files in parallel')
    parser.add_argument('--search_pattern',       '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',               '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    quality_predictor = ObjectiveMetricsPredictor(args.device)    
    scores = quality_predictor.predict_folder(args.input_dir_clean, args.input_dir_dist, args.batch_size, args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(scores, ofile, indent=4)


if __name__ == "__main__":
    main()
