from ssd_training_utils import create_model
from config import NUM_CLASSES, DEVICE, CLASSES, RESIZE_TO

from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
import cv2, os, time
from datetime import datetime

from tqdm.auto import tqdm


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

def video(show_detection_result, video_path):
    os.makedirs('inference_outputs/videos', exist_ok=True)

    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    detection_threshold = 0.2
    if video_path == None:
        video_name = "test_video2.mkv"
    else:
        video_name = video_path
    cap = cv2.VideoCapture(video_name)

    print(f'[INFO] Video path {video_name}')
    if (cap.isOpened() == False):
        print(f'[ERROR] Error while trying to read video. Please check path again {video_name}')
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    current_time = datetime.now()
    current_run = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    save_name = str(video_name).split(os.path.sep)[-1].split('.')[0] + '_' + current_run
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
        # frame = cv2.resize(frame, (RESIZE_TO, RESIZE_TO))

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
            if show_detection_result:
                cv2.imshow('Detection result', frame)
                cv2.waitKey(1)
            iterator+=1
            out.write(frame)
        else:
            break

    cap.release()

    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def pad_lists(preds, targets):
    max_length = max(len(preds), len(targets))
    padded_preds = preds + [0] * (max_length - len(preds))
    padded_targets = targets + [0] * (max_length - len(targets))
    return padded_preds, padded_targets


def get_metrics(test_loader, show_detection_result):
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

            labels_and_boxes = list(zip(labels, boxes))
            labels_and_boxes.sort(key=lambda x: (x[1][1], x[1][0]))  # Sort by ymin first, then xmin

            # Extract sorted labels
            sorted_input_labels = [label for label, box in labels_and_boxes]
            
            image_np = image.permute(1, 2, 0).cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect vahicules on input image
            _, detected_boxes_labels = detect_vahicule_on_image(image, model, DEVICE, CLASSES, show_detection_result)
            detected_boxes_labels.sort(key=lambda x: (x[1], x[0]))  # Sort by ymin first, then xmin

            sorted_detected_labels = [class_to_index[class_name] for _, _, _, _, class_name in detected_boxes_labels]
            sorted_detected_labels, sorted_input_labels = pad_lists(sorted_detected_labels, sorted_input_labels)
            
            all_targets.extend(sorted_input_labels)
            all_predictions.extend(sorted_detected_labels)
    

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    for i, class_name in enumerate(CLASSES):
        if i == 0:
            continue
        print(f"Class: {class_name}")
        print(f"\t Precision: {precision[i]}\n\t Recall: {recall[i]}\n\t F1 Score: {f1_score[i]}")
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)
    print(f"\nAverage metrics for each classes: \n\t Precision: {precision}\n\t Recall: {recall}\n\t F1 Score: {f1_score}\n")


def detect_vahicule_on_image(image, model, device, classes, show_detection_result, detection_threshold=0.25):
    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    model.to(device).eval()
    
    # Normalize image
    image_normalized = image.astype(np.float32) / 255.0
    
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float)
    image_input = torch.unsqueeze(image_input, 0)
    
    with torch.no_grad():
        outputs = model(image_input.to(device))
    
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    result_image = image.copy()
    detected_boxes_labels = []
    
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
    
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
    
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        num_of_cars = 0
    
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
    if show_detection_result:
        cv2.imshow('Detected objects', result_image)
        cv2.waitKey(1)
    return result_image, detected_boxes_labels

