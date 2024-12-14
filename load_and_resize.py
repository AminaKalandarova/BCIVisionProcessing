import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('C:/Users/amina/Desktop/Coding/Git/BCIVisionProcessing/BCIVisionProcessing/test_image.jpg', cv2.IMREAD_GRAYSCALE)

# Display the image using Matplotlib
plt.imshow(image, cmap='gray')
plt.title("Loaded Image")
plt.axis("off")  # Hide the axes
plt.show()

# Resize the image to 32x32
resized_image = cv2.resize(image, (32, 32))

# Display the resized image
plt.imshow(resized_image, cmap='gray')
plt.title("Resized Image (32x32)")
plt.axis("off")
plt.show()
