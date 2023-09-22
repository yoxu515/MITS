exp="mits"
config=$exp
exp_name="mits_R50_MITS"
ckpt_step=100000
ckpt_path="./pretrain_models/mits.pth"
gpu_num="4"

dataset="youtubevos2019"
split="val_all_frames"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step\
	--long_gap 10 --short_gap 3 --long_max 10
dataset="davis2017"
split="val"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step\
	--long_gap 10 --short_gap 3 --long_max 10

exp="mits_box"
config=$exp
exp_name="mits_R50_MITS"
ckpt_step=100000
ckpt_path="./pretrain_models/mits_box.pth"
gpu_num="4"

dataset="youtubevos2019"
split="val_all_frames"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step\
	--long_gap 10 --short_gap 3 --long_max 10
dataset="davis2017"
split="val"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path --ckpt_step $ckpt_step\
	--long_gap 10 --short_gap 3 --long_max 10


