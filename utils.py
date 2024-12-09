from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import requests
import cv2
from img2vec_pytorch import Img2Vec
from IPython.display import display




torch.hub._validate_ssl_certificates = False


# Function to load an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


# load image
def load_image_from_path(path):
    img = cv2.imread(path)
    # Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    pil_img = Image.fromarray(img)

    # Display the image
    # plt.imshow(pil_img)
    # plt.axis("off")  # Turn off axis numbers
    # plt.show()
    return pil_img


# Function to get the pre-trained Faster R-CNN model
def get_model():
    # Load a pre-trained model for object detection
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model


# TODO: test this
def resize_with_aspect_ratio(image, output_size=(224, 224), fill_color=(0, 0, 0)):
    # Calculate the target aspect ratio.
    target_aspect_ratio = output_size[0] / output_size[1]

    # Calculate the aspect ratio of the input image.
    img_aspect_ratio = image.width / image.height

    if img_aspect_ratio > target_aspect_ratio:
        # The image is wider than the target aspect ratio. Resize based on width.
        new_width = output_size[0]
        new_height = round(new_width / img_aspect_ratio)
    else:
        # The image is as wide or taller than the target aspect ratio. Resize based on height.
        new_height = output_size[1]
        new_width = round(new_height * img_aspect_ratio)

    # Resize the image with preserved aspect ratio.
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the desired output size, filled with the fill color (zero padding).
    final_image = Image.new("RGB", output_size, fill_color)
    paste_location = (
        (output_size[0] - new_width) // 2,
        (output_size[1] - new_height) // 2,
    )
    final_image.paste(resized_image, paste_location)

    return final_image


def detect_items(
    image,
    model=None,
    padding=15,
    output_size=(224, 224),
    showOutput=True,
    useLargestBird=False,
):
    img_tensor = F.to_tensor(image)

    if model is None:
        model = get_model()  # Ensure a model is always available

    original_image = image
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    selected_bird_box = None
    best_score = 0
    max_area = 0

    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        if label.item() == 16:  # Bird class ID in COCO
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            area = (xmax - xmin) * (ymax - ymin)

            if useLargestBird:
                # Choose the largest bird
                if area > max_area:
                    max_area = area
                    selected_bird_box = box.cpu().numpy()
                    best_score = score.item()
            else:
                # Choose the bird with the highest confidence score
                if score.item() > best_score:
                    best_score = score.item()
                    selected_bird_box = box.cpu().numpy()

    if selected_bird_box is not None:

        # Adjust the biggest bird box with padding
        padded_box = [
            selected_bird_box[0] - padding,
            selected_bird_box[1] - padding,
            selected_bird_box[2] + padding,
            selected_bird_box[3] + padding,
        ]

        # Crop the image based on the padded box
        cropped_image = image.crop(padded_box)

        # Resize the cropped image to the desired output size
        resized_image = cropped_image.resize(output_size, resample=Image.LANCZOS)

        if showOutput:
            # Create a copy of the original image for drawing
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)

            # Draw a rectangle around the biggest bird on the copy
            draw.rectangle(padded_box, outline="red", width=3)

            #Display the original image
            display(original_image)

            # Show the original image with the bounding box
            # image_copy.show()
            display(image_copy)

            # Show the cropped image
            # cropped_image.show()
            display(cropped_image)

            # Show the resized image
            display(resized_image)

    return resized_image


"""
Predicts the label of the image.
"""


def predict_image_label(image, model):
    # Convert image to array
    image_array = np.array(image)

    # Expand dimensions to match model input shape
    image_input = np.expand_dims(image_array, axis=0)

    # Make predictions
    predictions = model.predict(image_input)

    return predictions

def predict_image_label_cnn(image, model, labels):
    # Convert image to array
    image_array = np.array(image)

    # Expand dimensions to match model input shape
    image_input = np.expand_dims(image_array, axis=0)

    # Make predictions
    predictions = model.predict(image_input)

    # Get the predicted label
    predicted_label = labels[np.argmax(predictions[0])]

    return predicted_label


def image_to_vectors(image):
    img2vec = Img2Vec()
    img_vectors = img2vec.get_vec(image)
    return img_vectors


def preprocess_image(image_path):
    img = Image.open(image_path)
    return img


def extract_features(image, img2vec):
    img_features = img2vec.get_vec(image)
    return np.array(img_features).reshape(1, -1)


def predict_image_class(image_path, model, label_encoder):
    img2vec = Img2Vec()
    image = Image.open(image_path)
    detection_model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    detection_model.eval()

    detected_bird_image = detect_birds(image, detection_model, showOutput=False)
    if detected_bird_image is not None:
        image_features = extract_features(detected_bird_image, img2vec)
        prediction = model.predict(image_features)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)
        return predicted_label[0]
    return "No bird detected"


def predict_images(image_paths, model, label_encoder):
    results = []
    for image_path in image_paths:
        predicted_label = predict_image_class(image_path, model, label_encoder)
        results.append((image_path, predicted_label))
    return results
