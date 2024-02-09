D="data/crop_coco_0707/data_config.yaml"
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_std_96x1_b64_mix_for_pretrain_0707_data_config_2023-07-07_1000epc2/weights/best-int8.tflite" # int8 tflite model
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_std_96x1_b64_mix_for_pretrain_0707_data_config_2023-07-07_1000epc2/2023-07-07_15-58/avg/yolov5_667_p-0.8653_r-0.865_map50-0.6556-int8.tflite"
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R1_data_config_2023-07-10_1000epc2/2023-07-10_17-54/avg/yolov5_257_p-0.895_r-0.895_map50-0.7338-int8.tflite" 
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R2_data_config_2023-07-11_1000epc/2023-07-11_14-33/avg/yolov5_214_p-0.8925_r-0.8883_map50-0.6848-int8.tflite" 

#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R2_data_config_2023-07-11_1000epc/weights/best-int8.tflite" # int8 tflite model
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R3_data_config_2023-07-11_1000epc/weights/best-int8.tflite" # int8 tflite model
WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R3_data_config_2023-07-11_1000epc/2023-07-11_17-17/avg/yolov5_174_p-0.8911_r-0.8867_map50-0.6901-int8.tflite"

Channel=1
I=96

T=$(date +%Y-%m-%d)
IFS='/' read -r -a dtArray <<< "$D"
IFS='/' read -r -a modelFile <<< "$C"
#IFS='.' read -r -a modelName <<< "${modelFile[-1]}"

python val.py --device cpu \
	--data "${D}"\
	--weights "${WEIGHTS}" \
	--imgsz $I --imgch $Channel \
	--iou-thres 0.45 \
	--conf-thres 0 \
	--max-det 1\
