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
MODEL_ID = '3-uTCN-1000__causal__5-10-5__fraction-1.0-bs32__pb_dist'
MODEL_VERSION = 'latest'

class _Distortion_Service():

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

    def process(self, inputfile, drive_db, gpu=False, verbose=False):

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
        drives_db = [6, 12, 18, 24, 30]
        dummy = 1.0 # Try forcing this to be a float
        r_norm = float((int(drive_db) - min(drives_db)) / (max(drives_db) - min(drives_db)))

        # construct conditioning
        params = torch.tensor([dummy, drive_db])

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
        outfile += f"-pb_dist-{drive_db}-tcn3_out.wav"
        torchaudio.save(outfile, out.cpu(), 44100)

        return outfile

def Distortion_Service():
    """Factory function for Distortion_Service class.
    :return _Distortion_Service._instance (_Distortion_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Distortion_Service._instance is None:
        _Distortion_Service._instance = _Distortion_Service()
        _Distortion_Service.model = _Distortion_Service.load_model(_Distortion_Service._instance, MODEL_DIR, MODEL_ID, MODEL_VERSION)
    return _Distortion_Service._instance


def distort(inputfile, drive_db, gpu=False, verbose=False):
    distortion = Distortion_Service()

    return distortion.process(inputfile, drive_db, gpu, verbose)

if __name__ == "__main__":

    # Gradio Interface
    # Inputs: Audio, Slider, (Slider2)
    # Outputs: Audio
    iface = gr.Interface(fn=distort, 
                        inputs=[gr.inputs.Audio(type="filepath"),
                                gr.inputs.Slider(minimum=6, maximum=30, step=6, default=24),
                                ],
                        outputs=gr.outputs.Audio(type="file"))
    iface.launch(share=True)