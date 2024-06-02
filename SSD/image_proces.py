import torch
import cv2
import numpy as np

def detect_vahicule(image, model, device, classes, detection_threshold=0.25):
    
    COLORS = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    model.to(device).eval()
    
    # Preprocess the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_rgb /= 255.0
    image_input = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float)
    image_input = torch.unsqueeze(image_input, 0)
    
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(device))
    
    # Load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    result_image = image.copy()
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicted class names
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # Draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[classes.index(class_name)]
            cv2.rectangle(result_image,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          color[::-1],
                          3)
            cv2.putText(result_image,
                        class_name,
                        (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color[::-1],
                        2,
                        lineType=cv2.LINE_AA)
    
    return result_image

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Assuming 'model' is your pre-trained model and 'image_path' is the path to your image
    image_path = 'path/to/your/image.jpg'
    image = cv2.imread(image_path)
    model = create_model(num_classes=NUM_CLASSES, size=640)  # Define or load your model here
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES = ['background', 'vehicle']  # Define your classes here

    result_image = detect_vahicule(image, model, DEVICE, CLASSES)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    plt.imshow(result_image_rgb)
    plt.show()
