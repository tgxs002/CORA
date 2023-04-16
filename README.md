### Region Prompting
```shell
# run the following commands for region prompting
# COCO RN50
bash configs/COCO_RN50.sh exp_name 4 local
# COCO RN50x4
bash configs/COCO_RN50x4.sh exp_name 4 local
# LVIS RN50x4
bash configs/LVIS_RN50x4.sh exp_name 4 local
```
Note that you can also run it on a cluster with slurm by replacing local with slurm in the command.

### Exporting the region prompts
```shell
python export_rp.py --model_path /path/to/trained/model.pth --name output_name
```
