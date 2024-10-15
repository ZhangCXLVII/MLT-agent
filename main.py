import argparse
import sys, os
import random
import numpy as np
import logging
import traceback

import torch

from src import utils, engine, environ
from src.utils import ImageFeatures
from src.agent import build_agent


def setup(cfg, seed=2020):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # check vocab existence
    if not os.path.exists(cfg.TRAIN_VOCAB):
        utils.write_vocab(
            utils.build_vocab(splits=['train']), cfg.TRAIN_VOCAB)
    if not os.path.exists(cfg.TRAINVAL_VOCAB):
        utils.write_vocab(utils.build_vocab(
            splits=['train', 'val_seen', 'val_unseen']), cfg.TRAINVAL_VOCAB)


def main(args, cfg):
    logger = utils.get_main_logger(cfg.OUTPUT.LOG_DIR, cfg.MODEL.NAME)
    
    setup(cfg.DATA, seed=args.seed) # basic settings
    print(f"[1] random seed {args.seed} and vocab are setted, using config {args.config_file}")
    logger.info(f"[1] random seed {args.seed} and vocab are setted, using config {args.config_file}")

    logger.info("--------------------------")
    logger.info(cfg.TRAIN)
    logger.info("--------------------------")

    DEVICE = torch.device("cuda:{}".format(cfg.TRAIN.DEVICE) \
        if torch.cuda.is_available() else "cpu")
    print("[2] device {} is found... ".format(DEVICE))
    logger.info("[2] device {} is found... ".format(DEVICE))

    # create a batch training environment that will also preprocess text
    train_vocab = utils.read_vocab(cfg.DATA.TRAIN_VOCAB)
    tok = utils.Tokenizer(train_vocab, cfg.DATA.MAX_ENC_LEN)
    print("[3] vocab is loaded and tokenizer is assigned.")
    logger.info("[3] vocab is loaded and tokenizer is assigned.")
    
    # init R2R environment
    img_feature = ImageFeatures.read_in(cfg.DATA.IMG_FEAT_DIR)
    if cfg.DATA.NAME == "R2R":
        train_env = environ.R2RBatch(img_feature, cfg.TRAIN.BATCH_SIZE, splits=["train"], tokenizer=tok)
        valid_env = {
            "val_seen": environ.R2RBatch(img_feature, cfg.TRAIN.BATCH_SIZE, splits=["val_seen"], tokenizer=tok),
            "val_unseen": environ.R2RBatch(img_feature, cfg.TRAIN.BATCH_SIZE, splits=["val_unseen"], tokenizer=tok)
        }
    else: raise NotImplementedError
    print("[4] train and validation environment created.")
    logger.info("[4] train and validation environment created.") 

    # If you want to check whether the built-in code structure is right,
    # run this line:
    # engine.check_the_code(cfg, DEVICE, tok, valid_env)

    try:
        agent = build_agent(cfg, tok, DEVICE)
        # engine.train_model(cfg, agent, train_env, valid_env)
        if cfg.DATA.NAME in ["R2R", "RxR"]:
            trainer = engine.ClassicTrainer()
            logger.info("\t Using Classic Trainer ...")
            print("\t Using Classic Trainer ...")
            if cfg.TRAIN.EVAL_TRAIN:
                eval_train_env = {
                    f"round_{k}": environ.R2RBatch(
                        img_feature, cfg.TRAIN.BATCH_SIZE*2, splits=[f"train_round[{k}]_v3"], 
                        tokenizer=tok, data_name="CLR2R", data_dir="tasks/R2R-judy/data/CLR2Rv3")
                    for k in range(1, 6)
                }
            else: eval_train_env = None
        else: raise NotImplementedError
        trainer.train(cfg, agent, cfg.OUTPUT.TSBOARD_DIR, train_env, valid_env)
        # eval_train=cfg.TRAIN.EVAL_TRAIN, eval_train_env=eval_train_env
    except Exception:
        s = traceback.format_exc()
        print(s)
        logger.error(s)
    
    print("[5] Training Finished!")
    logger.info("[5] Training Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2R navigation Demo")
    parser.add_argument("--config-file", default="tasks/R2R-judy/configs/MLT-agent/MLT_config.yaml",
                        metavar="FILE",  help="path to config file",)
    parser.add_argument("--seed", default=2020, type=int, help="random seed")
    parser.add_argument("opts", help="Modify model config options using the command-line",
                        default=None, nargs=argparse.REMAINDER, )
    args = parser.parse_args()

    cfg = utils.get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze() # can not change it anymore

    main(args, cfg)


#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
