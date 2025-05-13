import easyocr
import deepl
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

## Deepl Translator
auth_key = "3998ee77-3ce9-4b7c-bff0-65db7106880b:fx"
deepl_client = deepl.Translator(auth_key)

## Image Path
image_path = './images/kor.png'
pil_image = Image.open(image_path)
cv2_image = cv2.imread(image_path)

## OCR Setup and Reader
reader = easyocr.Reader(['ko'], gpu=False) 
results = reader.readtext(image_path)

## Drawing on the image
draw = ImageDraw.Draw(pil_image)
font = ImageFont.truetype("arial.ttf", 24)


text_array = []
bbox_array = []
for bbox, text, prob in results:
    print(f"bbox: {bbox}, text: {text}, prob: {prob}")
    # Convert the NumPy arrays in bbox to tuples of integers for PIL drawing
    text_array.append(text), bbox_array.append(bbox)

    rectangle_coordinates = [tuple(map(int, bbox[0])), tuple(map(int, bbox[2]))]
    # draw.polygon(rectangle_coordinates, outline='red', width=2)
    cv2.rectangle(cv2_image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (255, 0, 0), 1)
    cv2.putText(cv2_image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# image.show()
# plt.imshow(cv2_image)
# plt.title("Image with Detected Text Bounding Boxes")
# plt.show()
    
# Translate the text using deepl api
translated_text = deepl_client.translate_text(text_array, target_lang="EN-US")

for i in range(len(text_array)):
    print(translated_text[i].text)  
    # Convert the NumPy arrays in the bbox to tuples of integers for PIL drawing
    top_left = tuple(map(int, bbox_array[i][0]))
    bottom_right = tuple(map(int, bbox_array[i][2]))

    # Define the coordinates for the rectangle
    rectangle_coordinates = [top_left,  bottom_right]

    # Draw the rectangle
    draw.rectangle(rectangle_coordinates, outline="red", fill="white", width=2)  # Draw the rectangle with a white fill
    draw.text(top_left, translated_text[i].text, fill="black", font=font)

# image.show()