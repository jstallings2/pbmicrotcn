import os
import glob
import time
import torch
import torchaudio
import gradio as gr

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel

# Make these constants or things that the user can change from the front end?
# Probably don't let user change these
MODEL_DIR = './api/lightning_logs/bulk'
MODEL_ID = '2-uTCN-300__causal__4-10-13__fraction-1.0-bs32__pb_verb'
MODEL_VERSION = 'latest'

class _Reverb_Service():

    model = None
    _instance = None

    def load_model(self, model_dir, model_id, model_version, gpu=False):
        # Select a specific version of this model to use
        print(model_dir, model_id)
        if model_version == 'latest':
            versions = os.listdir(os.path.join(model_dir,
                                                model_id,
                                                "lightning_logs"))
            version = max(versions)
        else:
            version = "version_" + model_version

        checkpoint_path = glob.glob(os.path.join(model_dir,
                                                model_id,
                                                "lightning_logs",
                                                version,
                                                "checkpoints",
                                                "*"))[0]


        hparams_file = os.path.join(model_dir, "hparams.yaml")
        batch_size = int(os.path.basename(model_id).split('-')[-1][2:].split('__')[0])
        model_type = os.path.basename(model_id).split('-')[1]
        epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

        map_location = "cuda:0" if gpu else "cpu"

        if model_type == "LSTM":
            model = LSTMModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location
            )

        else:
            model = TCNModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location
            )

        return model

    def process(self, inputfile, ratio, threshold, gpu=False, verbose=False):

        input, sr = torchaudio.load(inputfile, normalize=True) # watch for normalizing
        # input = input.float() / 32768

        # check if the input is mono
        if input.size(0) > 1:
            print(f"Warning: Model only supports mono audio, will downmix {input.size(0)} channels.")
            input = torch.sum(input, dim=0)

        # we will resample here if needed
        if sr != 44100:
            print(f"Warning: Model only operates at 44.1 kHz, will resample from {sr} Hz.")

        # Normalize params to [0,1]
        ratios = [2, 4, 8, 16, 32]
        thresholds = [10, 20, 30, 40, 50]
        r_norm = float((int(ratio) - min(ratios)) / (max(ratios) - min(ratios)))
        t_norm = float((int(threshold) - min(thresholds)) / (max(thresholds) - min(thresholds)))

        # construct conditioning
        params = torch.tensor([r_norm, t_norm])

        # add batch dimension
        input = input.view(1,1,-1)
        params = params.view(1,1,2)

        # move to GPU
        if gpu:
            input = input.to("cuda:0")
            params = params.to("cuda:0")
            self.model.to("cuda:0")

        # pass through model
        tic = time.perf_counter()
        out = self.model(input, params).view(1,-1)
        toc = time.perf_counter()
        elapsed = toc - tic

        if verbose:
            duration = input.size(-1)/44100
            print(f"Processed {duration:0.2f} sec in {elapsed:0.3f} sec => {duration/elapsed:0.1f}x real-time")

        # save output to disk (in same location)
        srcpath = os.path.dirname(inputfile)
        srcbasename = os.path.basename(inputfile).split(".")[0]
        outfile = os.path.join(srcpath, srcbasename)
        outfile += f"-pb_comp-{ratio}-{threshold}-tcn1_out.wav"
        torchaudio.save(outfile, out.cpu(), 44100)

        return outfile

def Reverb_Service():
    """Factory function for Reverb_Service class.
    :return _Reverb_Service._instance (_Reverb_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Reverb_Service._instance is None:
        _Reverb_Service._instance = _Reverb_Service()
        _Reverb_Service.model = _Reverb_Service.load_model(_Reverb_Service._instance, MODEL_DIR, MODEL_ID, MODEL_VERSION)
    return _Reverb_Service._instance

def verb(inputfile, room_size, wet_level, gpu=False, verbose=False):
    reverb = Reverb_Service()

    return reverb.process(inputfile, room_size, wet_level, gpu, verbose)

if __name__ == "__main__":

    # Gradio Interface
    # Inputs: Audio, Slider, Slider,
    # Outputs: Audio
    iface = gr.Interface(fn=verb, 
                        inputs=[gr.inputs.Audio(type="filepath"),
                                gr.inputs.Slider(minimum=0.2, maximum=1.0, step=0.2, default=0.8), 
                                gr.inputs.Slider(minimum=0.2, maximum=1.0, step=0.2, default=0.8),
                        ],
                        outputs=gr.outputs.Audio(type="file"))
    iface.launch(share=True)