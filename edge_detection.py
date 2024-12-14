import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform Otsu's thresholding and edge detection
def perform_edge_detection(image_path, output_path):
    # Load and resize the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("The specified image file could not be loaded. Check the file path.")
    resized_image = cv2.resize(image, (32, 32))

    # Apply Otsu's Thresholding
    otsu_threshold, binary_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate dynamic thresholds for Canny using Otsu's result
    lower_threshold = int(otsu_threshold * 0.5)  # Lower threshold as half of Otsu's threshold
    upper_threshold = int(otsu_threshold * 1.5)  # Upper threshold as 1.5 times Otsu's threshold

    print(f"Otsu Threshold: {otsu_threshold}")
    print(f"Canny Edge Detection Thresholds -> Lower: {lower_threshold}, Upper: {upper_threshold}")

    # Apply Canny Edge Detection using dynamic thresholds
    edges = cv2.Canny(resized_image, lower_threshold, upper_threshold)

    # Save the edge-detected image
    cv2.imwrite(output_path, edges)
    print(f"Edge-detected image saved at: {output_path}")

    # Display the original, binary, and edge-detected images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(resized_image, cmap='gray')
    plt.title("Resized Image (32x32)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image (Otsu's Threshold)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Edges Detected\n(Otsu's Threshold: {otsu_threshold:.2f})")
    plt.axis("off")

    plt.show()


# Main execution
if __name__ == "__main__":
    input_image_path = "test_image.jpg"  # Replace with your input image path
    output_edge_path = "edge_detected.jpg"  # Output path for edge-detected image

    perform_edge_detection(input_image_path, output_edge_path)
