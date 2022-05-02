import os,sys
import torch
import torchaudio
import argparse
import numpy as np
torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logging.basicConfig(format='%(message)s',
                            filename='rickshawrodeo.log',
                            filemode='w',
                            level=logging.DEBUG)


in_fs = 16000
in_stride = 0.02
in_stride_samps = in_stride * in_fs
in_window = 0.025
in_window_samps = in_window * in_fs
print('Input Stride:', in_stride_samps, 'samples, Input Window:', in_window_samps, 'samples.')


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, ignore):
        super().__init__()
        self.labels = labels
        self.ignore = ignore

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i not in self.ignore]
        return ''.join([self.labels[i] for i in indices])


if __name__ == '__main__':
    #NOTE: This will not work with pytorch later than 1.10.1.
    #      Known issues with pytorch 1.11.0
    #
    print(torch.__version__)
    print(torchaudio.__version__)
    print(device)

    parser = argparse.ArgumentParser(description="Input Audio File, Convert to 16ksps, Create Transcript using WAV2VEC2 model. Output file is <file>.transcript")
    parser.add_argument('-f', '--file', required=False, default="test.mp3", type=str, help='audio file.  see pytorchaudio docs for information about what is supported.')
    args = parser.parse_args()

    # SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    SPEECH_FILE = args.file

    if os.path.exists(SPEECH_FILE):
        logging.debug('Found file: %s', SPEECH_FILE)
        waveform, sample_rate = torchaudio.load(SPEECH_FILE)
        waveform = waveform.to(device)
    else:
        logging.debug('Did not find file: %s. Exiting.', SPEECH_FILE)
        sys.exit()

    #download the model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    #resample the audio file.
    if sample_rate != bundle.sample_rate:
        logging.debug('Changing sample rate from %g to %g', sample_rate, bundle.sample_rate)
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate

    print("Number of Samples:", waveform.size())
    print("Sample Rate:", sample_rate)
    print("Length of input in seconds:", waveform.size()[1] / sample_rate)
    print("Length of input in minutes:", waveform.size()[1] / sample_rate / 60.)
    print("Labels:", bundle.get_labels())
    model = bundle.get_model().to(device)



    #decode the data
    decoder = GreedyCTCDecoder(
        labels=bundle.get_labels(),
        ignore=(0, 1, 2, 3),
    )

    #pass in the waveform to the model (currently cropping to 40 seconds)
    emission = torch.empty((2, 0, 32), dtype=torch.float32)
    tmp_emission = torch.empty((2, 0, 32), dtype=torch.float32)
    N = 10*in_fs
    for i in range(4):
        inwav = waveform[:, i * N:(i * N) + N - 1]
        with torch.inference_mode():
            tmp_emission, _ = model(inwav)
            emission = torch.cat((emission, tmp_emission), 1)

    decoded_txt = decoder(emission[0])
    transcript = decoded_txt.replace('|', ' ')

    #save transcript to a file
    transcript_filename = SPEECH_FILE + '.transcript'
    fobj = open(transcript_filename,'w')
    logging.debug('Writing transcript %s to disk.', transcript_filename)
    fobj.write(transcript)

    '''
    print("Number of Samples:", waveform.size())
    print("Sample Rate:", bundle.sample_rate)
    print("Length of input in seconds:", waveform.size()[1] / bundle.sample_rate)
    print("Length of input in minutes:", waveform.size()[1] / bundle.sample_rate / 60.)
    print("Labels:", bundle.get_labels())
    model = bundle.get_model().to(device)

    '''


