import os
from .default import DefaultEngineConfig
# training MITS with GOT10k datasets
class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='R50_MITS',stage='default'):
        super().__init__(exp_name, model)
        if stage == 'default':
            self.STAGE_NAME = 'PRE_YTB_DAV'
        else:
            self.STAGE_NAME = stage
        if self.STAGE_NAME == 'PRE':
            self.DATASETS = ['static']

            self.DATA_DYNAMIC_MERGE_PROB = 1.0

            self.TRAIN_LR = 4e-4
            self.TRAIN_LR_MIN = 2e-5
            self.TRAIN_WEIGHT_DECAY = 0.03
            self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
            self.TRAIN_AUX_LOSS_RATIO = 0.1

            self.init_dir(data='./datasets',root='./data_wd/xyy/',eval='./')
            self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'
            self.PRETRAIN_MODEL = './resnet50-0676ba61.pth'
            # self.init_dir(data='../VOS02/datasets',root='./results',eval='./')
            # self.PRETRAIN_MODEL = '../VOS02/pretrain_models/resnet50-0676ba61.pth'

            self.DATA_PRE_STRONG_AUG = True
            self.DATA_TPS_PROB = 0.3
            self.DATA_TPS_SCALE = 0.02


        elif self.STAGE_NAME == 'PRE_YTB_DAV':

            self.DATASETS = ['got10k']
            self.init_dir(data='./datasets',root='./',eval='./')
            self.DATA_YTB_REPEAT = 4
            self.DATA_DAVIS_REPEAT = 15
            self.DATA_LASOT_REPEAT = 2
            self.DATA_DYNAMIC_MERGE_PROB_LASOT = 0.5
            self.DATA_DYNAMIC_MERGE_PROB_GOT10K = 0.5
            
            self.PRETRAIN_FULL = True  # if False, load encoder only
            self.PRETRAIN_MODEL = './pretrain_models/R50_DeAOTL_PRE.pth'

            
            self.TRAIN_SAVE_MED_STEP = 10000
            self.TRAIN_START_SAVE_MED_RATIO = 0.8

            self.TRAIN_BOXT = True
            self.TRAIN_TOTAL_STEPS = 100000
            self.TRAIN_BOXT_PROB = 0.3
            self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
            self.TRAIN_BOXT_MASK_LOSS_WEIGHT = 1.0
            self.TRAIN_BOXT_ID_LOSS_WEIGHT = 0.0
            
            self.MODEL_MS_ENCODER_EMBEDDING_DIMS = [256,256,256,128]

            self.DATA_RANDOM_GAUSSIAN_BLUR = 0.3
            self.DATA_RANDOM_GRAYSCALE = 0.2
            self.DATA_RANDOM_COLOR_JITTER = 0.8

            self.TRAIN_AUX_LOSS_WEIGHT = 1.0
            self.TRAIN_AUX_LOSS_RATIO = 0.75

            self.MODEL_USE_BOXT = True
            self.TRAIN_BOX_SIZE = (256,256)
            self.TRAIN_BOXT_PATCH_SIZE = 16
            self.MODEL_BOXT_NUM = 3
            self.TRAIN_BOXT_DROPPATH = 0.2
            self.TRAIN_BOXT_DROPPATH_CROSS = 0.2
            self.MODEL_BOXT_DUAL_CROSS = True
            
            self.MODEL_BOXT_DECODER = 'fpn'
            self.MODEL_BOXT_SHORTCUTS = False
            self.MODEL_BOXT_ID_SKIP = True

            self.MODEL_BOX_HEAD = 'pinpoint'
            self.MODEL_BOX_HEAD_DIM = 256
            self.MODEL_BOX_HEAD_LAYER = 3
            self.TRAIN_BOX_LOSS = True
            self.TRAIN_BOX_L1_LOSS_WEIGHT = 0.2
            self.TRAIN_BOX_GIOU_LOSS_WEIGHT = 0.1
            # self.TRAIN_BOX_FOCAL_LOSS_WEIGHT = 0.1
            # self.TEST_BOX_HEAD = True
            self.TEST_SAVE_BOX= True
            self.MODEL_BOX_HEAD_IN = 'feature'
            self.MODEL_BOX_HEAD_XY_POOLING = True
            self.MODEL_BOX_HEAD_XY_POOLING_TYPE = 'avg'
            self.MODEL_BOX_HEAD_NORM_ALIGN = True
            self.MODEL_BOX_HEAD_POOLING_PRECONV = True