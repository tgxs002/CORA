name=$0
. configs/controller.sh

args=" \
--coco_path $data_path \
--output_dir $work_dir \
--batch_size 4 \
--epochs 5 \
--lr_drop 4 \
--smca \
--backbone clip_RN50x4 \
--lr_backbone 0.0 \
--lr_language 0.0 \
--lr_prompt 1e-4 \
--text_len 25 \
--ovd \
--skip_encoder \
--attn_pool \
--region_prompt \
--roi_feat layer3 \
"

eval "$header$args$extra_args 2>&1 | tee -a $work_dir/exp_$now.txt"
