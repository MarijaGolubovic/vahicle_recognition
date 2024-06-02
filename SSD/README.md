# Vahicule detection by SSD model

[!IMPORTANT]
This isntructions is for Linux OS

### Installation
Download our source code:
```
git clone https://github.com/MarijaGolubovic/vahicle_recognition.git && cd vahicle_recognition/SSD
```

Create a python3 virtual environment ()
```
python3 -m venv pytorch_env && source activate pytorch_env/bin/activate
```
Install dependencies
```
pip3 install -r requirements.txt
```


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
```

### Validate model on video
```
 python3 ssd_detection.py --detect
```
If you want see detection result add argument `--show`.  Output video will be save in inference_outputs with suffix who presen current data and time. 

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

