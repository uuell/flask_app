import pytesseract
import cv2
from PIL import Image

# img = Image.open("testpaper-name&score.png")  # Replace with your image
# text = pytesseract.image_to_string(img)
# print(text)



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #  Path to tesseract.exe

image = cv2.imread("screenshot.png") # Load image with OpenCV

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # Apply thresholding to clean up the image

text = pytesseract.image_to_string(thresh) # OCR
print("Extracted Text:\n", text)

# lines = text.split('\n')
# name = score = "Not Found"

# for line in lines:
#     if "name" in line.lower():
#         name = line.split(":")[-1].strip()
#     if "score" in line.lower():
#         score = line.split(":")[-1].strip()

# print(f"\nName: {name}")
# print(f"Score: {score}")
