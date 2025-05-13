import cv2
import pytesseract

# OPTIONAL: Set path to tesseract.exe if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = cv2.imread('test_paper.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to clean background
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR
text = pytesseract.image_to_string(thresh)

print("Full Text:")
print(text)

# Optional: Extract name and score using simple string search
lines = text.split('\n')
name = score = "Not Found"

for line in lines:
    if "name" in line.lower():
        name = line.split(":")[-1].strip()
    if "score" in line.lower():
        score = line.split(":")[-1].strip()

print("\nExtracted:")
print(f"Name: {name}")
print(f"Score: {score}")
