import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image, ImageEnhance
import io
import time
import logging
import os  # Import os module

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set your Azure Computer Vision credentials
subscription_key = os.getenv("VISION_KEY", "8dae8f951a3449e384e4f7e851ebdcdd")
endpoint = os.getenv("VISION_ENDPOINT", "https://ai-faraahabdouai142514434982.cognitiveservices.azure.com/")

# Check if variables are set
if not subscription_key:
    st.error("Environment variable 'VISION_KEY' is not set.")
if not endpoint:
    st.error("Environment variable 'VISION_ENDPOINT' is not set.")

# Initialize Computer Vision client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Load models and processor
checkpoint = "openai/clip-vit-large-patch14"
model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

# Function to enhance image
def enhance_image(image):
    try:
        image = image.convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
        logging.info("Image enhancement completed.")
        return image_enhanced
    except Exception as e:
        logging.error(f"Error enhancing image: {e}")
        raise

# Function to classify image
def classify_image(image):
    try:
        image = enhance_image(image)
        candidate_labels = ["Gold", "Silver", "Platinum", "Palladium", "Diamond"]
        inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).numpy()
        scores = probs.tolist()
        results = [{"score": score, "label": candidate_label} for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])]
        logging.info(f"Raw classification results: {results}")
        confidence_threshold = 0.5
        filtered_results = [result for result in results if result['score'] > confidence_threshold]
        logging.info(f"Filtered classification results: {filtered_results}")
        return filtered_results, results
    except Exception as e:
        logging.error(f"Error classifying image: {e}")
        raise

# Function to analyze image with Azure Computer Vision
def analyze_image(image):
    try:
        image = enhance_image(image)
        image_stream = io.BytesIO()
        image.save(image_stream, format='JPEG')
        image_stream.seek(0)
        read_response = computervision_client.read_in_stream(image_stream, raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)
        text_results = []
        if read_result.status == OperationStatusCodes.succeeded:
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    bounding_box = {
                        "left": min(line.bounding_box[0::2]),
                        "top": min(line.bounding_box[1::2]),
                        "width": max(line.bounding_box[0::2]) - min(line.bounding_box[0::2]),
                        "height": max(line.bounding_box[1::2]) - min(line.bounding_box[1::2])
                    }
                    text_results.append({
                        "text": line.text,
                        "bounding_box": bounding_box
                    })
        logging.info(f"Text analysis results: {text_results}")
        return text_results
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        raise

# Streamlit app
st.title("Zeelan Image Classification and Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Enhancing and classifying the image...")
    filtered_results, results = classify_image(image)
    text_results = analyze_image(image)

    st.write("### Classification Results:")
    if filtered_results:
        for result in filtered_results:
            st.write(f"{result['label']}: {result['score']:.2f}")
    else:
        st.write("No high-confidence classification results found.")

    st.write("### Text Analysis Results:")
    st.write("Detected text:")
    if text_results:
        for text in text_results:
            st.write(f"{text['text']}")
    else:
        st.write("No text detected.")
