import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset

torch.backends.cudnn.benchmark = True

all_train_configs = [
    # {"name" : "uTCN-300",
    #  "model_type" : "tcn",
    #  "nblocks" : 4,
    #  "dilation_growth" : 10,
    #  "kernel_size" : 13,
    #  "causal" : True,
    #  "train_fraction" : 0.01,
    #  "batch_size" : 32
    # },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "TCN-300",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.10,
     "batch_size" : 32
    },
    {"name" : "LSTM-32",
     "model_type" : "lstm",
     "num_layers" : 1,
     "hidden_size" : 32,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 3,
     "dilation_growth" : 60,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1"
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 30,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
    },
    {"name" : "uTCN-324-16",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "channel_width" : 16,
    },
]

train_configs = [{"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    }]

n_configs = len(train_configs)

for idx, tconf in enumerate(train_configs):

    #if (idx+1) not in [14]: continue
    # if you only want to train a specific model

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='tcn', help='tcn or lstm') # --model_type='tcn'
    parser.add_argument('--root_dir', type=str, default='./data') # --root_dir='data/'
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='val')
    parser.add_argument('--train_length', type=int, default=65536)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=131072)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--effect', type=str, default='pb_comp', help='Available: pb_comp (default), pb_dist, pb_verb')

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    print(f"* Training config {idx+1}/{n_configs}")
    print(tconf)
  
    # let the model add what it wants
    if temp_args.model_type == 'tcn':
        parser = TCNModel.add_model_specific_args(parser)
    elif temp_args.model_type == 'lstm':
        parser = LSTMModel.add_model_specific_args(parser)

    # parse them args
    args = parser.parse_args()

    # set the seed
    pl.seed_everything(42)

    # only run 60 epochs
    args.max_epochs = 60

    # init the trainer and model 
    if tconf["model_type"] == 'tcn':
        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += "__causal" if tconf['causal'] else "__noncausal"
        specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
    elif tconf["model_type"] == 'lstm':
        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += f"__{tconf['num_layers']}-{tconf['hidden_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"

    if "max_epochs" in tconf:
        args.max_epochs = tconf["max_epochs"]
    else:
        args.max_epochs = 60

    if "train_loss" in tconf:
        args.train_loss = tconf["train_loss"]
        specifier += f"__loss-{tconf['train_loss']}"

    specifier += f"__{args.effect}"


    args.precision = 16

    args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
    print(args.default_root_dir)
    trainer = pl.Trainer.from_argparse_args(args)
    # The following line prevents overfitting by stopping if the val_loss is not improving after 3 validation epochs
    trainer.callbacks.append(EarlyStopping(monitor="val_loss", patience=3))

    # setup the dataloaders
    train_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                    subset=args.train_subset,
                                    fraction=tconf["train_fraction"],
                                    half=True if args.precision == 16 else False,
                                    preload=args.preload,
                                    length=args.train_length,
                                    effect=args.effect
                                    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=args.shuffle,
                                                batch_size=tconf["batch_size"],
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    val_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                    preload=args.preload,
                                    half=True if args.precision == 16 else False,
                                    subset=args.val_subset,
                                    length=args.eval_length,
                                    effect=args.effect)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                shuffle=False,
                                                batch_size=8,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    # create the model with args
    dict_args = vars(args)
    dict_args["nparams"] = 2 # NOTE: Currently changing this causes a bunch of matrix dim errors.

    if tconf["model_type"] == 'tcn':
        dict_args["nblocks"] = tconf["nblocks"]
        dict_args["dilation_growth"] = tconf["dilation_growth"]
        dict_args["kernel_size"] = tconf["kernel_size"]
        dict_args["causal"] = tconf["causal"]
        if "channel_width" in tconf:
            dict_args["channel_width"] = tconf["channel_width"]
        model = TCNModel(**dict_args)
    elif tconf["model_type"] == 'lstm':
        dict_args["num_layers"] = tconf["num_layers"]
        dict_args["hidden_size"] = tconf["hidden_size"]
        model = LSTMModel(**dict_args)

    # summary 
    torchsummary.summary(model, [(1,65536), (1,2)], device="cpu")


    # train!
    trainer.fit(model, train_dataloader, val_dataloader)
