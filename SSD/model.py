import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from albumentations.pytorch import ToTensorV2
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import classification_report

import torch
import cv2, time
import numpy as np
import os, pathlib
import glob as glob
import argparse

from xml.etree import ElementTree as et
import albumentations as A
import matplotlib.pyplot as plt

from tqdm.auto import tqdm



CLASSES = [
    '__background__', 'bus', 'car', 'truck'
]
NUM_CLASSES  = len(CLASSES)
RESIZE_TO = 640
NUM_EPOCHS = 1

OUT_DIR = 'outputs'
TRAIN_DIR  = "train"
VALID_DIR = "valid"
TEST_DIR = "test"

BATCH_SIZE = 5
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

plt.style.use('ggplot')

"""This class keeps track of the training and validation loss values
and helps to get the average for each epoch as well."""
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map
        
    def __call__(
        self, 
        model, 
        current_valid_map, 
        epoch, 
        OUT_DIR,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, f"{OUT_DIR}/best_model.pth")

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def save_model(epoch, model, optimizer):
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')

def save_loss_plot(
    OUT_DIR, 
    train_loss_list, 
    x_label='iterations',
    y_label='train loss',
    save_name='train_loss'
):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')

def save_mAP(OUT_DIR, map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-', 
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")


class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image and the annotaions
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # Left corner x-coordinates.
            xmin = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            xmax = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            ymin = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            ymax = int(member.find('bndbox').find('ymax').text)
            

            if xmax <= xmin or ymax <= ymin:
                print(xmax, xmin, ymax, ymin)
                print(image_name)
            
            # Resized image `width`, `height` to desired size
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

def create_train_dataset(DIR):
    train_dataset = CustomDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform()
    )
    return train_dataset

def create_valid_dataset(DIR):
    valid_dataset = CustomDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    return valid_loader


# if __name__ == '__main__':
#     dataset = CustomDataset(
#         TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
#     )
    
#     # function to visualize a single sample
#     def visualize_sample(image, target):
#         for box_num in range(len(target['boxes'])):
#             box = target['boxes'][box_num]
#             label = CLASSES[target['labels'][box_num]]
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             cv2.rectangle(
#                 image, 
#                 (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
#                 (0, 0, 255), 
#                 2
#             )
#             cv2.putText(
#                 image, 
#                 label, 
#                 (int(box[0]), int(box[1]-5)), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.7, 
#                 (0, 0, 255), 
#                 2
#             )
#         cv2.imshow('Image', image)
#         cv2.waitKey(0)

def create_model(num_classes=91, size=300):

    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )
    
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    print(in_channels)
    
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

def video():
    os.makedirs('inference_outputs/videos', exist_ok=True)

    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    detection_threshold = 0.2

    video_name = "test_video2.mkv"
    cap = cv2.VideoCapture(video_name)

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    save_name = str(video_name).split(os.path.sep)[-1].split('.')[0] + "najnoviji"
    print(save_name)
    
    out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    frame_count = 0 
    total_fps = 0
    iterator = 0
    # Read until end of video.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            image = frame.copy()
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Make the pixel range between 0 and 1.
            image /= 255.0
            # Bring color channels to front (H, W, C) => (C, H, W).
            image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image_input = torch.tensor(image_input, dtype=torch.float)
            image_input = torch.unsqueeze(image_input, 0)
            start_time = time.time()
            # Predictions
            with torch.no_grad():
                outputs = model(image_input.to(DEVICE))
            end_time = time.time()
            
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                # Filter out boxes according to `detection_threshold`.
                boxes = boxes[scores >= 0.25].astype(np.int32)
                draw_boxes = boxes.copy()
                # Get all the predicited class names.
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
                
                # Draw the bounding boxes and write the class name on top of it.
                for j, box in enumerate(draw_boxes):
                    class_name = pred_classes[j]
                    color = COLORS[CLASSES.index(class_name)]
                    # Recale boxes.
                    xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                    ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                    xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                    ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                    cv2.rectangle(frame,
                            (xmin, ymin),
                            (xmax, ymax),
                            color[::-1], 
                            3)
                    cv2.putText(frame, 
                                class_name, 
                                (xmin, ymin-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, 
                                color[::-1], 
                                2, 
                                lineType=cv2.LINE_AA)
            cv2.putText(frame, f"{fps:.0f} FPS", 
                        (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
            print("------------------", iterator)
            iterator+=1
            out.write(frame)
        else:
            break

    cap.release()

    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

def create_test_dataset(DIR):
    test_dataset = CustomDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return test_dataset

def create_test_loader(test_dataset, num_workers=0):
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    return test_loader

def get_labels_from_test_loader_ex(test_loader):
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])


    for images, targets in test_loader:
        for image, target in zip(images, targets):
            labels = target['labels'].tolist()
            print("labele",labels)

            image_np = image.permute(1, 2, 0).cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            result_image = detect_vahicule(image, model, DEVICE, CLASSES)

            cv2.imshow("sjjddd", result_image)
            cv2.waitKey(0)


def pad_tensors(preds, targets):
    max_length = max(len(preds), len(targets))
    padded_preds = torch.nn.functional.pad(preds, (0, max_length - len(preds)))
    padded_targets = torch.nn.functional.pad(targets, (0, max_length - len(targets)))
    return padded_preds, padded_targets

def pad_lists(preds, targets):
    max_length = max(len(preds), len(targets))
    padded_preds = preds + [0] * (max_length - len(preds))
    padded_targets = targets + [0] * (max_length - len(targets))
    return padded_preds, padded_targets

def get_f1_score(preds, targets):
    
    targets = torch.tensor(targets)
    preds = torch.tensor(preds)

    preds_, targets_  = pad_tensors(preds, targets)

    f1_score = multiclass_f1_score(preds_, targets_, num_classes=NUM_CLASSES, average='weighted')

    return f1_score

def get_labels_from_test_loader_old(test_loader):
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    sorted_detected_labels_all_images = []
    sorted_input_labels_all_images = []
    class_to_index = {'bus': 1, 'car': 2, 'truck': 3}

    number_of_samples = 0
    f1_sum = 0

    class_names = ['background', 'bus', 'car', 'truck']
    
    for images, targets in test_loader:
        for image, target in zip(images, targets):
            labels = target['labels'].tolist()
            boxes = target['boxes'].tolist()

            # Combine labels and boxes
            labels_and_boxes = list(zip(labels, boxes))

            # Sort combined labels and boxes by (ymin, xmin)
            labels_and_boxes.sort(key=lambda x: (x[1][1], x[1][0]))  # Sort by ymin first, then xmin

            # Extract sorted labels
            sorted_input_labels = [label for label, box in labels_and_boxes]
            
            # sorted_input_labels_all_images.append(sorted_input_labels)

            image_np = image.permute(1, 2, 0).cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            result_image, detected_boxes_labels = detect_vahicule(image, model, DEVICE, CLASSES)

            # Sort detected boxes and labels by (ymin, xmin)
            detected_boxes_labels.sort(key=lambda x: (x[1], x[0]))  # Sort by ymin first, then xmin

            sorted_detected_labels = [class_to_index[class_name] for _, _, _, _, class_name in detected_boxes_labels]
            # sorted_detected_labels_all_images.append(sorted_detected_labels)

            # targets_ = torch.tensor(sorted_input_labels)
            # preds_ = torch.tensor(sorted_detected_labels)

            # preds_, targets_  = pad_tensors(preds_, targets_)


            f1_score = get_f1_score(sorted_detected_labels, sorted_input_labels)
            f1_sum +=f1_score
            number_of_samples +=1
            
            print(f"F1 Score: {f1_score}")
            
            # print("F1 scores for each class:", f1_scores)

            # cv2.imshow("sjjddd", result_image)
            # cv2.waitKey(0)
    print("F1 score for test dataset:  ", f1_sum/number_of_samples)
    # return sorted_input_labels_all_images, sorted_detected_labels_all_images


from sklearn.metrics import precision_recall_fscore_support

def get_labels_from_test_loader(test_loader):
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_targets = []
    all_predictions = []
    class_to_index = {'bus': 1, 'car': 2, 'truck': 3}

    for images, targets in test_loader:
        for image, target in zip(images, targets):
            labels = target['labels'].tolist()
            boxes = target['boxes'].tolist()

            # Combine labels and boxes
            labels_and_boxes = list(zip(labels, boxes))

            # Sort combined labels and boxes by (ymin, xmin)
            labels_and_boxes.sort(key=lambda x: (x[1][1], x[1][0]))  # Sort by ymin first, then xmin

            # Extract sorted labels
            sorted_input_labels = [label for label, box in labels_and_boxes]
            
            image_np = image.permute(1, 2, 0).cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            _, detected_boxes_labels = detect_vahicule(image, model, DEVICE, CLASSES)

            # Sort detected boxes and labels by (ymin, xmin)
            detected_boxes_labels.sort(key=lambda x: (x[1], x[0]))  # Sort by ymin first, then xmin

            sorted_detected_labels = [class_to_index[class_name] for _, _, _, _, class_name in detected_boxes_labels]
            sorted_detected_labels, sorted_input_labels = pad_lists(sorted_detected_labels, sorted_input_labels)
            all_targets.extend(sorted_input_labels)
            all_predictions.extend(sorted_detected_labels)
    
    
    # Compute precision, recall, and F1 score for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, labels=list(range(NUM_CLASSES)), average=None)
    for i, class_name in enumerate(CLASSES):
        if i == 0:
            continue
        print(f"Class: {class_name}")
        print(f"\t Precision: {precision[i]}\n\t Recall: {recall[i]}\n\t F1 Score: {f1_score[i]}")
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, labels=list(range(NUM_CLASSES)), average='weighted')
    print(f"\nAverage metrics for each classes: \n\t Precision: {precision}\n\t Recall: {recall}\n\t F1 Score: {f1_score}\n")


def detect_vahicule_ex(image, model, device, classes, detection_threshold=0.25):
    
    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    # image = cv2.imread("/home/marija/vahicle_recognition/SSD/test/1711098625-4207957_jpg.rf.4b36df8d93989e88b54d77f6170c3b44.jpg")
    
    model.to(device).eval()
    
    # Normalize image
    image_normalized = image.astype(np.float32) / 255.0
    
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float)
    image_input = torch.unsqueeze(image_input, 0)
    
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(device))
    
    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    result_image = image.copy()
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicted class names.
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        num_of_cars = 0
        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            num_of_cars+=1
            color = COLORS[classes.index(class_name)]
            # Rescale boxes.
            xmin = int((box[0] / image.shape[1]) * result_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * result_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * result_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * result_image.shape[0])
            cv2.rectangle(result_image,
                          (xmin, ymin),
                          (xmax, ymax),
                          color[::-1], 
                          3)
            cv2.putText(result_image, 
                        class_name, 
                        (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        color[::-1], 
                        2, 
                        lineType=cv2.LINE_AA)
            return result_image
        print(num_of_cars)
    return result_image


def detect_vahicule(image, model, device, classes, detection_threshold=0.25):
    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    model.to(device).eval()
    
    # Normalize image
    image_normalized = image.astype(np.float32) / 255.0
    
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float)
    image_input = torch.unsqueeze(image_input, 0)
    
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(device))
    
    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    result_image = image.copy()
    detected_boxes_labels = []
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicted class names.
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        num_of_cars = 0
        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            num_of_cars += 1
            color = COLORS[classes.index(class_name)]
            # Rescale boxes.
            xmin = int((box[0] / image.shape[1]) * result_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * result_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * result_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * result_image.shape[0])
            cv2.rectangle(result_image,
                          (xmin, ymin),
                          (xmax, ymax),
                          color[::-1], 
                          3)
            cv2.putText(result_image, 
                        class_name, 
                        (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        color[::-1], 
                        2, 
                        lineType=cv2.LINE_AA)
            detected_boxes_labels.append((xmin, ymin, xmax, ymax, class_name))
    cv2.imshow('Detected objects', result_image)
    cv2.waitKey(0)
    return result_image, detected_boxes_labels


if __name__ == '__main__':
    test_dataset = create_test_dataset(TEST_DIR)
    test_loader = create_test_loader(test_dataset, NUM_WORKERS)

    get_labels_from_test_loader(test_loader)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--detect', 
        default=False,
        type=bool,
        help='run training mode'
    )

    parser.add_argument('--train', type=bool, default=False, help='detection mode')

    args, _ = parser.parse_known_args()
    print(args.detect, type(args.detect), args)
    evaluation = args.detect
    print(evaluation, type(evaluation), args)
    if True:
        video()
    else:
        os.makedirs('outputs', exist_ok=True)
        train_dataset = create_train_dataset(TRAIN_DIR)
        valid_dataset = create_valid_dataset(VALID_DIR)
        train_loader = create_train_loader(train_dataset, NUM_WORKERS)
        valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}\n")

        
        model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
        model = model.to(DEVICE)
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=0.0005
        )
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
        )

        train_loss_hist = Averager()
        # To store training loss and mAP values.
        train_loss_list = []
        map_50_list = []
        map_list = []

        MODEL_NAME = 'model'
        save_best_model = SaveBestModel()

        # Training loop.
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

            train_loss_hist.reset()

            start = time.time()
            train_loss = train(train_loader, model)
            metric_summary = validate(valid_loader, model)
            print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
            print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
            print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
            print(metric_summary)

            train_loss_list.append(train_loss)
            map_50_list.append(metric_summary['map_50'])
            map_list.append(metric_summary['map'])

            save_best_model(
                model, float(metric_summary['map']), epoch, 'outputs'
            )

            save_model(epoch, model, optimizer)

            save_loss_plot(OUT_DIR, train_loss_list)

            save_mAP(OUT_DIR, map_50_list, map_list)
            scheduler.step()
