import easyocr
from googletrans import Translator

# Initialize the EasyOCR reader
reader = easyocr.Reader(['ko'], detector='DB', recognizer='Transformer', gpu=False)  # Specify the language(s) you want to recognize

# Path to the image file
image_path = 'kor.png'

# Perform text recognition
results = reader.readtext(image_path)

# Print the results
for (bbox, text, confidence) in results:
    print(f"Detected text: {text} (Confidence: {confidence:.2f})")