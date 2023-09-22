import importlib
import sys

sys.path.append('.')
sys.path.append('..')

import torch
import torch.multiprocessing as mp

from networks.managers.evaluator import Evaluator


def main_worker(gpu, cfg, seq_queue=None, info_queue=None, enable_amp=False):
    # Initiate a evaluating manager
    evaluator = Evaluator(rank=gpu,
                          cfg=cfg,
                          seq_queue=seq_queue,
                          info_queue=info_queue)
    # Start evaluation
    if enable_amp:
        with torch.cuda.amp.autocast(enabled=True):
            evaluator.evaluating()
    else:
        evaluator.evaluating()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval VOS")
    parser.add_argument('--exp_name', type=str, default='default')

    parser.add_argument('--config', type=str, default='pre')
    # parser.add_argument('--model', type=str, default='aott')
    parser.add_argument('--lstt_num', type=int, default=-1)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--split', type=str, default='')

    parser.add_argument('--ema', action='store_true')
    parser.set_defaults(ema=False)

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--inflip', action='store_true')
    parser.set_defaults(inflip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--long_gap', type=int, default=5)
    parser.add_argument('--short_gap', type=int, default=1)
    parser.add_argument('--long_max', type=int, default=9999)
    parser.add_argument('--save_prob',action='store_true')
    parser.set_defaults(save_prob=False)
    parser.add_argument('--save_prob_scale',type=float,default=0.5)
    parser.set_defaults(amp=False)
    parser.add_argument('--top_k',type=float,default=-1)
    parser.add_argument('--box_ref',action='store_true')
    parser.add_argument('--box_crop',action='store_true')
    parser.add_argument('--box_crop_keep_size',action='store_true')
    parser.add_argument('--box_crop_ratio',type=float,default=2.)
    parser.add_argument('--box_crop_size',type=int,default=384)
    parser.add_argument('--box_crop_thr',type=int,default=500)
    parser.add_argument('--box_crop_skip',action='store_true')
    parser.add_argument('--box_head',action='store_true')
    parser.add_argument('--box_mem_skip',action='store_true')
    parser.add_argument('--box_mem_skip_long',action='store_true')
    parser.add_argument('--box_mem_skip_short',action='store_true')
    parser.add_argument('--box_mem_skip_thr',type=float,default=0.8)
    parser.add_argument('--box_weight_mask',action='store_true')
    parser.add_argument('--box_gauss_scale',type=float,default=1.0)
    parser.add_argument('--box_head_backup',action='store_true')
    parser.add_argument('--box_mem_backup',action='store_true')
    parser.add_argument('--box_head_mask',action='store_true')
    parser.add_argument('--box_head_mask_weight',type=float,default=0.0)
    parser.add_argument('--box_head_mask_mem',action='store_true')
    

    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.config)
    cfg = engine_config.EngineConfig(args.exp_name)

    cfg.TEST_EMA = args.ema

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = args.gpu_num

    cfg.TEST_SAVE_PROB = args.save_prob
    cfg.TEST_SAVE_PROB_SCALE = args.save_prob_scale
    cfg.TEST_TOP_K = args.top_k
    
    if args.long_gap != 5:
        cfg.TEST_LONG_TERM_MEM_GAP = args.long_gap
    if args.short_gap != 1:
        cfg.TEST_SHORT_TERM_MEM_GAP = args.short_gap
    if args.long_max != 9999:
        cfg.TEST_LONG_TERM_MEM_MAX = args.long_max

    if args.lstt_num > 0:
        cfg.MODEL_LSTT_NUM = args.lstt_num

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset

    if args.split != '':
        cfg.TEST_DATASET_SPLIT = args.split

    cfg.TEST_FLIP = args.flip
    cfg.TEST_INPLACE_FLIP = args.inflip
    cfg.TEST_MULTISCALE = args.ms

    if cfg.TEST_MULTISCALE != [1.]:
        cfg.TEST_MAX_SHORT_EDGE = args.max_resolution  # for preventing OOM
    else:
        cfg.TEST_MAX_SHORT_EDGE = None  # the default resolution setting of CFBI and AOT
    cfg.TEST_MAX_LONG_EDGE = args.max_resolution * 800. / 480.

    # box ref
    if args.box_ref:
        if cfg.MODEL_USE_BOXT:
            cfg.TEST_BOXT = True
        elif cfg.MODEL_USE_ID_ENCODER:
            cfg.TEST_BOX_ID_ENCODER = True
        cfg.TEST_SAVE_BOX = True
    if args.box_head:
        cfg.TEST_BOX_HEAD = True
    cfg.TEST_BOX_CROP = args.box_crop
    cfg.TEST_BOX_CROP_RATIO = args.box_crop_ratio
    cfg.TEST_BOX_CROP_KEEP_SIZE = args.box_crop_keep_size
    cfg.TEST_BOX_CROP_SIZE = (args.box_crop_size,args.box_crop_size)
    cfg.TEST_BOX_CROP_THR = args.box_crop_thr
    cfg.TEST_BOX_CROP_SKIP = args.box_crop_skip
    cfg.TEST_BOX_MEM_SKIP = args.box_mem_skip
    cfg.TEST_BOX_MEM_SKIP_LONG = args.box_mem_skip_long
    cfg.TEST_BOX_MEM_SKIP_SHORT = args.box_mem_skip_short
    cfg.TEST_BOX_MEM_SKIP_THR = args.box_mem_skip_thr
    cfg.TEST_BOX_WEIGHTED_MASK = args.box_weight_mask
    cfg.TEST_BOX_GAUSS_SCALE = args.box_gauss_scale
    cfg.TEST_BOX_HEAD_BACKUP = args.box_head_backup
    cfg.TEST_BOX_BACKUP_MEM = args.box_mem_backup
    cfg.TEST_BOX_HEAD_MASK = args.box_head_mask
    cfg.TEST_BOX_HEAD_MASK_WEIGHT = args.box_head_mask_weight
    cfg.TEST_BOX_HEAD_MASK_MEM = args.box_head_mask_mem
    
    
    
    if args.gpu_num > 1:
        mp.set_start_method('spawn')
        seq_queue = mp.Queue()
        info_queue = mp.Queue()
        mp.spawn(main_worker,
                 nprocs=cfg.TEST_GPU_NUM,
                 args=(cfg, seq_queue, info_queue, args.amp))
    else:
        main_worker(0, cfg, enable_amp=args.amp)


if __name__ == '__main__':
    main()
