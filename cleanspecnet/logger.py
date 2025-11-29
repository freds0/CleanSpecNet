import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Disables interactivity

class Logger:
    """
    Logger class to handle TensorBoard logging.
    """

    def __init__(self, log_dir, rank=0):
        """
        Initializes the SummaryWriter.

        Parameters:
        - log_dir (str): Directory where TensorBoard logs will be saved.
        - rank (int): Rank of the process (for distributed training). Only rank 0 will log.
        """
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def add_scalar(self, tag, scalar_value, global_step=None):
        """
        Logs a scalar value.

        Parameters:
        - tag (str): Data identifier.
        - scalar_value (float): Value to record.
        - global_step (int, optional): Global step value to record.
        """
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_figure(self, tag, figure, global_step=None):
        """
        Logs a matplotlib figure.

        Parameters:
        - tag (str): Data identifier.
        - figure (matplotlib.figure.Figure): Figure to log.
        - global_step (int, optional): Global step value to record.
        """
        if self.writer:
            self.writer.add_figure(tag, figure, global_step)

    def add_audio(self, tag, waveform, global_step=None, sample_rate=16000):
        """
        Logs an audio waveform.

        Parameters:
        - tag (str): Data identifier.
        - waveform (torch.Tensor): Audio waveform tensor.
        - global_step (int, optional): Global step value to record.
        - sample_rate (int): Sample rate of the waveform.
        """
        if self.writer:
            self.writer.add_audio(tag, waveform, global_step, sample_rate)

    def close(self):
        """
        Closes the SummaryWriter.
        """
        if self.writer:
            self.writer.close()
