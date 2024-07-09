CONFIG=D:/mmdetection-master-win/configGoggle2/deformable_detr_r50_16x2_50e_coco_base_loss2_r02_clsconv.py
CHECKPOINT=D:/mmdetection-master-win/work_dirs_goggle2/deformable_detr_r50_16x2_50e_coco_base_loss2_r02_clsconv/epoch_50.pth
REPEAT_NUM=1
MAX_ITER=2000
LOG_INTERVAL=50
D:/mmdetection-master-win/venv/Scripts/python.exe -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} \
tools/analysis_tools/benchmark.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    [--repeat-num ${REPEAT_NUM}] \
    [--max-iter ${MAX_ITER}] \
    [--log-interval ${LOG_INTERVAL}] \
    --launcher pytorch