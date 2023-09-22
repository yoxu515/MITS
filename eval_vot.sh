exp="mits"
config=$exp
exp_name="mits_R50_MITS"
gpu_num="4"
ckpt_step=100000
ckpt_path="./pretrain_models/mits.pth"

dataset="lasot"
# dataset="trackingnet"
split="test"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--long_gap 30 --short_gap 10 --long_max 10 --box_ref --box_head\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step


exp="mits_box"
config=$exp
exp_name="mits_R50_MITS"
gpu_num="4"
ckpt_step=100000
ckpt_path="./pretrain_models/mits_box.pth"

dataset="lasot"
# dataset="trackingnet"
split="test"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--long_gap 30 --short_gap 10 --long_max 10 --box_ref --box_head\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step


exp="mits_got"
config=$exp
gpu_num="4"
ckpt_step=100000
ckpt_path="./pretrain_models/mits_got.pth"

dataset="got10k"
split="test"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--long_gap 10 --short_gap 2 --long_max 10 --box_ref --box_head\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step