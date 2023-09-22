import os
import importlib


class DefaultEngineConfig():
    def __init__(self, exp_name='default', model='aott'):
        model_cfg = importlib.import_module('configs.models.' +
                                            model.lower()).ModelConfig()
        self.__dict__.update(model_cfg.__dict__)  # add model config

        self.EXP_NAME = exp_name + '_' + self.MODEL_NAME

        self.STAGE_NAME = 'YTB'

        self.DATASETS = ['youtubevos']
        self.DATA_WORKERS = 4 #8
        self.DATA_RANDOMCROP = (465,
                                465) if self.MODEL_ALIGN_CORNERS else (464,
                                                                       464)
        self.DATA_RANDOMFLIP = 0.5
        self.DATA_MAX_CROP_STEPS = 10
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_MIN_SCALE_FACTOR = 0.7
        self.DATA_MAX_SCALE_FACTOR = 1.3
        
        self.DATA_PRE_STRONG_AUG = False # for PRE
        self.DATA_TPS_PROB = 0.0
        self.DATA_TPS_SCALE = 0.0
        self.DATA_RANDOM_GAUSSIAN_BLUR = 0.0 #0.3
        self.DATA_RANDOM_GRAYSCALE = 0.0 #0.2
        self.DATA_RANDOM_COLOR_JITTER = 0.0 #0.8
        
        self.DATA_RANDOM_REVERSE_SEQ = True
        self.DATA_SEQ_LEN = 5
        self.DATA_DAVIS_REPEAT = 5
        self.DATA_YTB_REPEAT = 1
        self.DATA_LASOT_REPEAT = 3
        self.DATA_RANDOM_GAP_DAVIS = 12  # max frame interval between two sampled frames for DAVIS (24fps)
        self.DATA_RANDOM_GAP_YTB = 3  # max frame interval between two sampled frames for YouTube-VOS (6fps)
        self.DATA_RANDOM_GAP_BL30K = 12
        self.DATA_RANDOM_GAP_VIP = 3
        self.DATA_RANDOM_GAP_LASOT = 3
        self.DATA_RANDOM_GAP_GOT10K = 5
        self.DATA_DYNAMIC_MERGE_PROB = 0.3
        self.DATA_DYNAMIC_MERGE_PROB_BL30K = 0.0
        self.DATA_DYNAMIC_MERGE_PROB_VIP = 0.1
        self.DATA_DYNAMIC_MERGE_PROB_LASOT = 0.0
        self.DATA_DYNAMIC_MERGE_PROB_GOT10K = 0.0
        self.DATA_LASOT_BOX_LABEL = False
        self.DATA_GOT10K_BOX_LABEL = False

        self.DATA_YTB_BALANCE_SAMPLE = False
        self.DATA_YTB_BALANCE_RATIO = 0.0
        self.DATA_YTB_USE_VOSP = False

        self.PRETRAIN = True
        self.PRETRAIN_FULL = False  # if False, load encoder only
        self.PRETRAIN_MODEL = ''
        self.PRETRAIN_ID_MODEL = ''
        # self.PRETRAIN_MODEL = './data_wd/pretrain_model/mobilenet_v2.pth'
        # self.PRETRAIN_MODEL = './pretrain_models/mobilenet_v2-b0353104.pth'

        self.TRAIN_TOTAL_STEPS = 100000
        self.TRAIN_START_STEP = 0
        self.TRAIN_WEIGHT_DECAY = 0.07
        self.TRAIN_WEIGHT_DECAY_EXCLUSIVE = {
            # 'encoder.': 0.01
        }
        self.TRAIN_WEIGHT_DECAY_EXEMPTION = [
            'absolute_pos_embed', 'relative_position_bias_table',
            'relative_emb_v', 'conv_out'
        ]
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5 if 'mobilenetv2' in self.MODEL_ENCODER else 1e-5
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_LR_ENCODER_RATIO = 0.1
        self.TRAIN_LR_WARM_UP_RATIO = 0.05
        self.TRAIN_LR_COSINE_DECAY = False
        self.TRAIN_LR_RESTART = 1
        self.TRAIN_LR_UPDATE_STEP = 1
        self.TRAIN_AUX_LOSS_WEIGHT = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 1.0
        self.TRAIN_BCE_LOSS_WEIGHT = 0.5
        self.TRAIN_IOU_LOSS_WEIGHT = 0.5
        self.TRAIN_WITH_BOX = False
        self.TRAIN_OPT = 'adamw'
        self.TRAIN_SGD_MOMENTUM = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCH_SIZE = 16
        self.TRAIN_TBLOG = True
        self.TRAIN_TBLOG_STEP = 50
        self.TRAIN_TBLOG_IMG_STEP = 1000
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_IMG_LOG = False
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15
        self.TRAIN_SEQ_TRAINING_FREEZE_PARAMS = ['patch_wise_id_bank','id_encoder','id_post_conv','box_id','BoxT']
        self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
        self.TRAIN_HARD_MINING_RATIO = 0.5
        self.TRAIN_EMA_RATIO = 0.1
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 1000
        self.TRAIN_SAVE_MED_STEP = 10000
        self.TRAIN_START_SAVE_MED_RATIO = 1.0
        self.TRAIN_MAX_KEEP_CKPT = 8
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_DATASET_FULL_RESOLUTION = False
        self.TRAIN_ENABLE_PREV_FRAME = False
        self.TRAIN_ENCODER_FREEZE_AT = 2
        self.TRAIN_LSTT_EMB_DROPOUT = 0.
        self.TRAIN_LSTT_ID_DROPOUT = 0.
        self.TRAIN_LSTT_DROPPATH = 0.1
        self.TRAIN_LSTT_DROPPATH_SCALING = False
        self.TRAIN_LSTT_DROPPATH_LST = False
        self.TRAIN_LSTT_LT_DROPOUT = 0.
        self.TRAIN_LSTT_ST_DROPOUT = 0.

        self.TEST_GPU_ID = 0
        self.TEST_GPU_NUM = 1
        self.TEST_FRAME_LOG = False
        self.TEST_DATASET = 'youtubevos'
        self.TEST_DATASET_FULL_RESOLUTION = False
        self.TEST_DATASET_SPLIT = 'val'
        self.TEST_CKPT_PATH = None
        # if "None", evaluate the latest checkpoint.
        self.TEST_CKPT_STEP = None
        self.TEST_FLIP = False
        self.TEST_INPLACE_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MAX_SHORT_EDGE = None
        self.TEST_MAX_LONG_EDGE = 800 * 1.3
        self.TEST_WORKERS = 4
        self.TEST_SAVE_PROB = False
        self.TEST_SAVE_PROB_SCALE = 0.5
        self.TEST_SAVE_LOGIT = False
        self.TEST_BOX_FILTER = False
        self.TEST_TOP_K = -1

        self.TEST_INTERMEDIATE_PRED = False
        self.TRAIN_INTERMEDIATE_PRED_LOSS = False

        self.TRAIN_BOX_ID_ENCODER_PROB = 0.0
        self.TRAIN_BOX_ID_ENCODER_ONLY = False
        self.TRAIN_BOX_ID_ENCODER_MASK_LOSS_WEIGHT = 0.0
        self.TRAIN_BOX_ID_ENCODER_ID_LOSS_WEIGHT = 0.0
        self.TEST_BOX_ID_ENCODER = False
        
        self.TRAIN_BOXT = False
        self.TRAIN_BOXT_ONLY = False
        self.TRAIN_BOXT_MASK_ONLY = False
        self.TRAIN_BOXT_PROB = 0.0
        self.TRAIN_BOXT_PATCH_SIZE = 8
        self.TRAIN_BOX_SIZE = (128,128)
        self.TRAIN_BOXT_DROPPATH = 0.1
        self.TRAIN_BOXT_DROPPATH_CROSS = False
        self.TRAIN_BOXT_MASK_LOSS_WEIGHT = 0.0
        self.TRAIN_BOXT_ID_LOSS_WEIGHT = 0.0
        self.TRAIN_BOXT_MASK_PRJ_LOSS_WEIGHT = 0.0
        self.TEST_BOXT = False

        self.TRAIN_BOX_LOSS = False
        self.TRAIN_BOX_L1_LOSS_WEIGHT = 0.0
        self.TRAIN_BOX_GIOU_LOSS_WEIGHT = 0.0
        self.TRAIN_BOX_FOCAL_ALPHA = 2.0
        self.TRAIN_BOX_FOCAL_BETA = 4.0
        self.TRAIN_BOX_FOCAL_LOSS_WEIGHT = 0.0
        self.TRAIN_BOX_CTR_LOSS_WEIGHT = 0.0
        self.TEST_BOX_HEAD = False
        
        
        self.TEST_BOX_CROP = False
        self.TEST_BOX_CROP_RATIO = 4
        self.TEST_BOX_CROP_DYNAMIC = False
        self.TEST_BOX_CROP_SIZE = (384,384)
        self.TEST_BOX_CROP_KEEP_SIZE = True
        self.TEST_BOX_CROP_THR = 500
        self.TEST_BOX_CROP_SKIP = False
        self.TEST_SAVE_BOX = False
        self.TEST_BOX_MEM_SKIP = False
        self.TEST_BOX_MEM_SKIP_THR = 0.8
        self.TEST_BOX_MEM_SKIP_LONG = False
        self.TEST_BOX_MEM_SKIP_SHORT = False
        self.TEST_BOX_HEAD_BACKUP = False
        self.TEST_BOX_BACKUP_MEM = False
        self.TEST_BOX_WEIGHTED_MASK = False
        self.TEST_BOX_GAUSS_SCALE = 1.0
        self.TEST_BOX_CROP_BOXH = False
        self.TEST_BOX_HEAD_MASK = False
        self.TEST_BOX_HEAD_MASK_MEM = False
        self.TEST_BOX_HEAD_MASK_WEIGHT = 0.0

        # GPU distribution
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "nccl"  # "gloo"
        self.DIST_URL = "tcp://127.0.0.1:13241"
        self.DIST_START_GPU = 0

    def init_dir(self,data='./datasets',root='./results',eval=None):
        self.DIR_DATA = data
        self.DIR_DAVIS = os.path.join(self.DIR_DATA, 'DAVIS')
        self.DIR_YTB = os.path.join(self.DIR_DATA, 'YTB')
        self.DIR_STATIC = os.path.join(self.DIR_DATA, 'Static')
        self.DIR_BL30K = os.path.join(self.DIR_DATA,'BL30K')
        self.DIR_LASOT = os.path.join(self.DIR_DATA,'LaSOT')
        self.DIR_GOT10K = os.path.join(self.DIR_DATA,'GOT10K')
        self.DIR_LASOTTest = os.path.join(self.DIR_DATA, 'LaSOTTest')
        self.DIR_GOT10KTest = os.path.join(self.DIR_DATA, 'GOT10KTest')
        self.DIR_TRACKINGNETTest = os.path.join(self.DIR_DATA, 'TrackingNetTest')

        self.DIR_ROOT = root

        self.DIR_RESULT = os.path.join(self.DIR_ROOT, 'result', self.EXP_NAME,
                                       self.STAGE_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_AUX = os.path.join(self.DIR_RESULT, 'aux')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_MED_CKPT = os.path.join(self.DIR_RESULT, 'med_ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        
        if eval == None:
            self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')
        else:
            self.DIR_EVAL = os.path.join(eval,'result',self.EXP_NAME,self.STAGE_NAME)
            self.DIR_EVALUATION = os.path.join(self.DIR_EVAL,'eval')

        for path in [
                self.DIR_RESULT, self.DIR_CKPT, self.DIR_EMA_CKPT,
                self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG,
                self.DIR_TB_LOG, self.DIR_MED_CKPT
        ]:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except Exception as inst:
                    print(inst)
                    print('Failed to make dir: {}.'.format(path))
