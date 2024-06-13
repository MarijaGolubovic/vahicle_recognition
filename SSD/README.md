# Vahicule detection by SSD model

>[!IMPORTANT]
>This instructions is for Linux OS

## Installation
Download our source code:
```
git clone https://github.com/MarijaGolubovic/vahicle_recognition.git && cd vahicle_recognition/SSD
```

Create a python3 virtual environment
```
python3 -m venv pytorch_env && source pytorch_env/bin/activate
```
Install dependencies
```
pip3 install -r requirements.txt
```

### Dataset
##### Small Dataset:
    * Train set: 309 images
    * Valid set: 88 images
    * Test set: 43 images

[Download small dataset](https://app.roboflow.com/carstracksbus/cars-ljnwr/1)

### TO DO
** Add large dataset **


### Walkthrough
```
.
|---- config.py             # default parameter configuration 
|---- custom_dataset.py     # contains function for data preparation
|---- ssd_training_utils.py # contains function for traing and model preparation
|---- validation_utils.py   # contains function for validation and vehicle detection on image/video
|---- ssd_detection.py      # main function
|---- requirements.txt      
|---- outputs/
        |--- best_model.pth # program load this model
        |--- last_model.pth
|---- train/
|---- valid/
|---- test/
|---- video*.mp4
```


<br/>

### Validate model on video
Add path video  to function `video()` inside `validation_utils.py`.
```
 python3 ssd_detection.py --detect
```
If you want see detection result add argument `--show`.  Output video will be save in 'inference_outputs/videos' with suffix who presen current data and time. 

### Evaluate trained model on test data set

```
    python3 ssd_detection.py --eval
```
If you want see detection result add argument `--show`

### Train own model
```
python3 ssd_detection.py --train
```
If you want change number of epochs add argument `--epochs num_epochs`. Training result will be saved in directory with current data time. If you want validate trained model should be copy best_model.pth in `outpusts/` directory.

<br/><br/>


## Performance Metrics

| Class   | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| bus     | 0.5234    | 0.4621 | 0.4908   |
| car     | 0.9206    | 0.8491 | 0.8834   |
| truck   | 0.1667    | 0.2000 | 0.1818   |

<br/>

**Average Metrics for All Classes:**
- Precision: 0.4027
- Recall: 0.3778
- F1 Score: 0.3890

To ensure a balanced consideration of performance across all classes during calculation, adjust the 'average' parameter from  `'macro'`  to  `'weighted'`  within the  `get_metrics`  function.
