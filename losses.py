import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def naive_loss_fn(clean_audio, denoised_audio, clean_spec, denoised_spec):
    loss_audio = F.mse_loss(denoised_audio, clean_audio)
    loss_spec = F.l1_loss(denoised_spec, clean_spec)
    return loss_audio + loss_spec


class CleanSpecNetLoss(nn.Module):
    """
    Loss function for CleanSpecNet based on CleanUNet 2 paper.

    Minimizes L1 distance between log-magnitude spectrograms:
    L_spec = (1/T_spec) * || log(1 + y) - log(1 + y_hat) ||_1

    Where y is the target (clean) spectrogram magnitude and y_hat is the predicted.
    """
    def __init__(self):
        super().__init__()

    def forward(self, clean_spec, predicted_spec):
        """
        Args:
            clean_spec: Target clean spectrogram magnitude (B, F, T)
            predicted_spec: Predicted spectrogram magnitude (B, F, T)

        Returns:
            L1 loss between log-compressed spectrograms, normalized by T_spec
        """
        # Apply log compression: log(1 + x)
        log_clean = torch.log1p(clean_spec)
        log_predicted = torch.log1p(predicted_spec)

        # L1 loss normalized by time steps (T_spec)
        # F.l1_loss with reduction='mean' divides by all elements (B * F * T)
        # To match the paper's (1/T_spec), we use sum over freq and mean over time
        T_spec = clean_spec.size(-1)
        loss = torch.sum(torch.abs(log_clean - log_predicted)) / T_spec

        # Normalize by batch size as well for stable training
        loss = loss / clean_spec.size(0)

        return loss


class CleanUnetLoss():
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        self.ell_p = ell_p
        self.ell_p_lambda = ell_p_lambda
        self.stft_lambda = stft_lambda
        self.mrstftloss = mrstftloss

    def __call__(self, clean_audio, denoised_audio):
        B, C, L = clean_audio.shape
        output_dic = {}
        loss = 0.0

        # Reconstruction loss (L1 or L2)
        if self.ell_p == 2:
            ae_loss = F.mse_loss(denoised_audio, clean_audio)
        elif self.ell_p == 1:
            ae_loss = F.l1_loss(denoised_audio, clean_audio)
        else:
            raise NotImplementedError(f"ell_p={self.ell_p} is not supported. Use 1 (L1) or 2 (L2).")

        loss += ae_loss * self.ell_p_lambda
        output_dic["reconstruct"] = ae_loss.item() * self.ell_p_lambda

        # STFT-based losses
        if self.stft_lambda > 0:
            sc_loss, mag_loss = self.mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
            loss += (sc_loss + mag_loss) * self.stft_lambda
            output_dic["stft_sc"] = sc_loss.item() * self.stft_lambda
            output_dic["stft_mag"] = mag_loss.item() * self.stft_lambda

        return loss, output_dic


class CleanUNet2Loss:
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        self.cleanunet_loss = CleanUnetLoss(ell_p, ell_p_lambda, stft_lambda, mrstftloss)

    def __call__(self, clean_audio, denoised_audio):
        loss_cleanunet, _ = self.cleanunet_loss(clean_audio, denoised_audio)

        # ⚠️ Remover esta linha se já estiver usando L1 na CleanUnetLoss
        loss_l1 = F.l1_loss(clean_audio, denoised_audio, reduction='mean')

        return loss_cleanunet + loss_l1  # ou apenas `return loss_cleanunet`


def stft(x, fft_size, shift_size, win_length, window):
    window = window.to(x.device)
    x_stft = torch.stft(
        x, n_fft=fft_size, hop_length=shift_size, win_length=win_length,
        window=window, return_complex=True, center=True
    )
    return x_stft


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(torch.clamp(y_mag, min=1e-7)), torch.log(torch.clamp(x_mag, min=1e-7)))


class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", band="full"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window).abs()
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window).abs()

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2
            x_mag = x_mag[:, freq_mask_ind:, :]
            y_mag = y_mag[:, freq_mask_ind:, :]

        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        sc_lambda=0.1,
        mag_lambda=0.1,
        band="full"
    ):
        super().__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, hs, wl, window, band)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x, y):
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))
            y = y.view(-1, y.size(2))

        sc_loss = 0.0
        mag_loss = 0.0
        for stft_loss in self.stft_losses:
            sc_l, mag_l = stft_loss(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss = sc_loss * self.sc_lambda / len(self.stft_losses)
        mag_loss = mag_loss * self.mag_lambda / len(self.stft_losses)

        return sc_loss, mag_loss
