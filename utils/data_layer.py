"""
This package contains Neural Modules responsible for ASR-related
data layers.
"""
__all__ = ['AudioToTextDataLayer']
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from .dataset import (AudioDataset, IterableDataset, seq_collate_fn)
from .features import WaveformFeaturizer

def pad_to(x, k=8):
    """Pad int value up to divisor of k.
    Examples:
        >>> pad_to(31, 8)
        32
    """
    return x + (x % k > 0) * (k - x % k)
class AudioDataLayer(IterableDataset):

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

class AudioToTextDataLayer(nn.Module):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (str): Dataset parameter.
            End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    def __init__(
            self, *,
            manifest_filepath,
            labels,
            batch_size,
            sample_rate=16000,
            int_values=False,
            bos_id=None,
            eos_id=None,
            pad_id=None,
            min_duration=0.1,
            max_duration=None,
            normalize_transcripts=True,
            trim_silence=False,
            load_audio=True,
            drop_last=False,
            shuffle=True,
            num_workers=4,
            placement='cpu',
            # perturb_config=None,
            **kwargs
    ):
        super().__init__()

        self._featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None)

        # Set up dataset
        dataset_params = {'manifest_filepath': manifest_filepath,
                          'labels': labels,
                          'featurizer': self._featurizer,
                          'max_duration': max_duration,
                          'min_duration': min_duration,
                          'normalize': normalize_transcripts,
                          'trim': trim_silence,
                          'bos_id': bos_id,
                          'eos_id': eos_id,
                          'logger': None,
                          'load_audio': load_audio}

        self._dataset = AudioDataset(**dataset_params)

        # Set up data loader
        if placement == 'cuda':
            print('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

