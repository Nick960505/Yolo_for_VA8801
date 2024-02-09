D="/home/lawr/workspace/human_detect_project/yolov5/data/crop_coco_0707/R3_data_config.yaml"
#D="/home/lawr/workspace/human_detect_project/yolov5/data/mix_for_pretrain_0707/data_config.yaml"
C="/home/lawr/workspace/human_detect_project/yolov5/models/yolov5n_WM01.yaml"
H="lh_hyp_scratch_tiny.yaml"
#WEIGHTS="/home/vianne/yolov5_2023/yolov5/runs/train/exp30/weights/best.pt" # for transfer learning
WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_std_96x1_b64_mix_for_pretrain_0707_data_config_2023-07-07_1000epc2/weights/best.pt"
#WEIGHTS=""
Channel=1
I=96
B=64
E=1000
T=$(date +%Y-%m-%d)
IFS='/' read -r -a dtArray <<< "$D"
IFS='/' read -r -a modelFile <<< "$C"
IFS='.' read -r -a modelName <<< "${modelFile[-1]}"
IFS='.' read -r -a DataName <<< "${dtArray[-1]}"

if [ -z "$WEIGHTS" ]; then
    w="std"
else
    w="pretrain"
fi

python train.py --workers 8 --device 0\
	--data "${D}"\
	--weights "${WEIGHTS}" \
	--hyp data/hyps/"${H}" \
	--name ${modelName[0]}_${w}_${I}x${Channel}_b${B}_${dtArray[-2]}_${DataName[0]}_${T}_${E}epc\
	--batch-size $B \
	--cfg "${C}" \
	--img-size $I --img-ch $Channel \
	--epochs $E \
	--patience 500
	# --freeze 10 # for transfer learning

