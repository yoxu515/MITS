stage="pre_ytb_dav"
exp="mits"
gpu_ids='0,1,2,3'

config=$exp
python tools/train.py --amp \
	--exp_name ${exp} \
	--config ${config} \
	--gpu_ids ${gpu_ids} \
    --stage 'PRE_YTB_DAV'