#SOURCE="/home/lawr/workspace/human_detect_project/datasets/inference_images/cropped_coco/train/all/background/"
#SOURCE="/home/lawr/workspace/human_detect_project/datasets/inference_images/cropped_coco/train/all/human/"
#SOURCE="/home/lawr/workspace/human_detect_project/datasets/inference_images/cropped_cctv/human/"
SOURCE="/home/lawr/workspace/human_detect_project/datasets/inference_images/96x96ROI_Data/human/"
#WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R1_data_config_2023-07-10_1000epc2/weights/best-int8.tflite" # int8 tflite model
WEIGHTS="/home/lawr/workspace/human_detect_project/yolov5/runs/train/yolov5n_WM01_pretrain_96x1_b64_crop_coco_0707_R2_data_config_2023-07-11_1000epc/2023-07-11_14-33/avg/yolov5_214_p-0.8925_r-0.8883_map50-0.6848-int8.tflite"
Channel=1
I=96

T=$(date +%Y-%m-%d)
IFS='/' read -r -a dtArray <<< "$SOURCE"
IFS='/' read -r -a modelFile <<< "$C"
#IFS='.' read -r -a modelName <<< "${modelFile[-1]}"

if [[ "${SOURCE}" == */human/ ]]; then
    cls=0
elif [[ "${SOURCE}" == */background/ ]]; then
    cls=1
else
    cls=99
fi

python tflite_int8_infer_conf_human_detect_excel.py -d 1 \
	-s "${SOURCE}"\
	-w "${WEIGHTS}" \
	--img-size $I --img_ch $Channel \
	--conf_thres 0.0 \
	--store_image_result \
	--show_cls_conf \
	--cls $cls \


