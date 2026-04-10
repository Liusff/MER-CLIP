# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import glob

_PROJECT_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), '..')

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, osp.join(_PROJECT_ROOT, 'projects', 'actionclip'))
from mmaction.registry import RUNNERS

#CUDA_VISIBLE_DEVICES=2,3
def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--loso', type=str, default="casme3", help="decide subject list")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

def find_best_acc_file(folder_path):
    for file in os.listdir(folder_path):
        if "best_acc" in file:
            return os.path.join(folder_path, file)
    return None

def main(args, train_subset, val_subset):

    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)
    cfg.train_dataloader['dataset']['subset'] = train_subset
    cfg.val_dataloader['dataset']['subset'] = val_subset
    cfg.test_dataloader['dataset']['subset'] = val_subset
    cfg.test_evaluator['save_path'] = osp.join(cfg.work_dir, "results.csv")
    cfg.work_dir = osp.join(cfg.work_dir, f"testsub_{val_subset[0]}/")

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        print("build")
        runner = RUNNERS.build(cfg)
    print("RNNERS_BUILD end!!!, strat train!")
    # start training
    runner.train()
    best_ckpt = find_best_acc_file(cfg.work_dir)
    if best_ckpt is not None:
        print("Tesing using ckpt: ", best_ckpt)
        '''
        cfg.load_from = best_ckpt
        cfg.work_dir = osp.join(cfg.work_dir, "evaluation/")
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir)
        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)
        '''
        runner._load_from = best_ckpt
        runner.test()
    else:
        print("Path have no best ckpt: ", cfg.work_dir)



if __name__ == '__main__':
    args = parse_args()
    #main(args)
    if args.loso == "samm":
        subject_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,28,30,31,32,33,34,35,37]
        print("Perform {}_fold LOSO for SAMM...".format(len(subject_list)))
        for sub in subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "samm_3cls":
        subject_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,28,30,31,32,33,34,35,37]
        val_subject_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,28,30,31,32,33,34,35,37]
        print("Perform {}_fold LOSO for SAMM...".format(len(val_subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "samm_3cls2":
        subject_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,28,30,31,32,33,34,35,37]
        val_subject_list = [19,20,21,22,23,26,28,30,31,32,33,34,35,37]
        print("Perform {}_fold LOSO for SAMM...".format(len(val_subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "casme2":
        subject_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]  #
        print("Perform {}_fold LOSO for CASME^2...".format(len(subject_list)))
        for sub in subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "casme2_3cls":
        subject_list = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26]  #
        val_subject_list = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26]
        print("Perform {}_fold LOSO for CASME^2...".format(len(subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "casme3": #1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 39, 40, 41, 42, 77, 138, 139,\
                         #142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159,\
                         #160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,\
                         #178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,\
                         #196, 197, 198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217
        subject_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 39, 40, 41, 42, 77, 138, 139,\
                         142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159,\
                         160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,\
                         178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,\
                         196, 197, 198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217]  # 94
        val_subject_list = [173, 174, 175, 176, 177,\
                         178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,\
                         196, 197, 198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217]
        print("Perform {}_fold LOSO for CASME^3...".format(len(val_subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "casme3_3cls": #1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 39, 40, 41, 42, 77, 138, 139,\
                         #142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159,\
                         #160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,\
                         #178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,\
                         #196, 197, 198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217
        subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 39, 40, 41, 42, 77, 138, 139, 142,\
                        144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163,\
                        165, 166, 167, 169, 170, 171, 172, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,\
                        186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 200, 201, 202, 203, 204, 206, 207,\
                        208, 209, 210, 211, 212, 213, 214, 215, 216, 217]  # 88
        val_subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 39, 40, 41, 42, 77, 138, 139, 142,\
                        144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163,\
                        165, 166, 167, 169, 170, 171, 172, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,\
                        186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 200, 201, 202, 203, 204, 206, 207,\
                        208, 209, 210, 211, 212, 213, 214, 215, 216, 217]
        print("Perform {}_fold LOSO for CASME^3...".format(len(subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "mmew":
        subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  #
        val_subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        print("Perform {}_fold LOSO for MMEW...".format(len(subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
    elif args.loso == "mmew2":
        subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  #
        val_subject_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        print("Perform {}_fold LOSO for MMEW...".format(len(subject_list)))
        for sub in val_subject_list:
            val_subset = [sub]
            train_subset = list(set(subject_list)-set(val_subset))
            print("Train Subset: ",train_subset)
            print("Test Subset: ",val_subset)
            main(args, train_subset, val_subset)
#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]