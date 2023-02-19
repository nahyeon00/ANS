import torch
import argparse
from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer, seed_everything
import yaml
import os
from ANSclassifier import *
from ng_dataprocess import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                       type=int,
                       default=0)
    parser.add_argument('--data_path',
                        type=str,
                        default='/workspace/intent/newADB/data/',
                        help='where to prepare data')
    parser.add_argument("--dataset", 
                        default='stackoverflow', 
                        type=str, 
                        help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio",
                        default=0.5,
                        type=float,
                        help="The number of known classes")
    parser.add_argument("--labeled_ratio",
                        default=1.0,
                        type=float,
                        help="The ratio of labeled samples in the training set")
    parser.add_argument('--max_epoch',
                       type=int,
                        default=10,
                       help='maximum number of epochs to train')
    parser.add_argument('--num_gpus',
                       type=int,
                       default=1,
                       help='number of available gpus')
    parser.add_argument('--ckpt_path',
                       type=str,
                       default = '/workspace/intent/ANS',
                       help='checkpoint file path')
    parser.add_argument('--model_save_path',
                       type=str,
                       default='ANSmodel',
                       help='where to save checkpoint files')
    parser.add_argument('--max_seq_len',
                       type=int,
                       default=45,
                       help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')
    parser.add_argument('--device',
                       type=int,
                       default=0,
                       help='batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='num of worker for dataloader')
    parser.add_argument('--output_dir',
                       type=str,
                       default='/workspace/intent/ANS/results/',
                       help='output dir')
    parser.add_argument("--results_file_name", 
                        type=str,
                        default = 'results.csv', 
                        help="The file name of all the results.")
    parser.add_argument("--num_labels", 
                        type=int, 
                        default = 0, 
                        help="known label list + unseen label")
    parser.add_argument("--mode", 
                        type=str, 
                        default = 'ANS', 
                        help="for test dataloader")
    parser.add_argument("--noise_grad_step", 
                        type=int, 
                        default = 3, 
                        help="gradient step for gradient ascend")
    parser.add_argument("--radius", 
                        type=int, 
                        default = 8, 
                        help="radius")
    parser.add_argument("--gamma", 
                        type=int, 
                        default = 0.5, 
                        help="radius")
                



    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    seed_everything(args.seed, workers=True)
    
    dm = ANSDataModule(args = args)

    dm.setup('fit')
    
    model = ANS(args)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_save_path,
        filename='{ANS:02d}-{val_acc:.3f}',
        verbose=True,
        save_last=False,
        mode='min',
        save_top_k=1,
    )
    # early_stopping = EarlyStopping(
    #     monitor='val_acc', 
    #     mode='max',
    # )

    
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=[1],
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)

    # print("finish train")
    # model.freeze()
    # model.eval()
    # print("start test")
    # trainer.test(model, dm)
    # print("finish test")
    # print(test_results)
    # save_results(args, test_results)

    
if __name__ == '__main__':
    main()
