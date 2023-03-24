# Prepare Datasets

We provide instruction for preparing datasets.

## COCO

Please download the [COCO dataset](https://cocodataset.org/#download), unzip it, and make sure it is in the following structure:

```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

Then, download `instances_train2017_base.json`, (`instances_train2017_base_RN50relabel.json`, `instances_train2017_base_RN50x4relabel_pre.json`) and `instances_val2017_basetarget.json` from [this Google Drive](https://drive.google.com/drive/folders/1kDOch_Rh7o2mPOSD39F2HJ88aPOEh_qm?usp=share_link). They are used for region prompting, localizer training and evaluation respectively. Please put then under `coco/annotations`.

Export the dataset path (path/to/coco) by executing: 
```
export data_path='path/to/coco'
```

## LVIS

<!-- export lvis_path='~/datasets/lvis' -->

TBD.

 <!-- [LVIS dataset](https://www.lvisdataset.org/dataset)
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v1_{train,val}.json
  lvis_v1_image_info_test{,_challenge}.json
```

Since the folder `lvis/` is large in size, you could soft link it in dataset directory. For example, run `ln -s DIR_to_LVIS datasets/lvis`.

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
``` -->
