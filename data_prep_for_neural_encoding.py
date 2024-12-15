import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Function to process and prepare image data for neural encoding
def prepare_image_for_neural_encoding(image_path, output_path, new_size=(64, 64)):
    # Load and resize the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("The specified image file could not be loaded. Check the file path.")
    resized_image = cv2.resize(image, new_size)

    # Apply Otsu's Thresholding
    otsu_threshold, binary_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate dynamic thresholds for Canny using Otsu's result
    lower_threshold = int(otsu_threshold * 0.5)  # Lower threshold as half of Otsu's threshold
    upper_threshold = int(otsu_threshold * 1.5)  # Upper threshold as 1.5 times Otsu's threshold

    # Apply Canny Edge Detection using dynamic thresholds
    edges = cv2.Canny(resized_image, lower_threshold, upper_threshold)

    # Visualize the edge-detected image
    plt.imshow(edges, cmap='gray')
    plt.title("Edge-Detected Image")
    plt.axis('off')
    plt.show()

    # Normalize the edge-detected image to the range [0, 1]
    print(f"Edge image min/max before normalization: {np.min(edges)}, {np.max(edges)}")
    normalized_image = edges / 255.0  # Convert pixel values to [0, 1]
    print(f"Edge image min/max after normalization: {np.min(normalized_image)}, {np.max(normalized_image)}")

    # Flatten the image data into a 1D array (for neural encoding)
    flattened_data = normalized_image.flatten()

    # Preview the flattened data
    print(f"Flattened Data (First 100 values): {flattened_data[:100]}")

    # Save the edge-detected image
    cv2.imwrite(output_path, edges)
    print(f"Edge-detected image saved at: {output_path}")

    # Save flattened data as a .csv file
    with open('flattened_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flattened_data)
    print("Flattened data saved as 'flattened_data.csv'")

    # Return the prepared data for neural encoding
    return flattened_data, otsu_threshold, lower_threshold, upper_threshold

# Main execution
if __name__ == "__main__":
    input_image_path = "test_image.jpg"  # Replace with your input image path
    output_edge_path = "edge_detected.jpg"  # Output path for edge-detected image

    # Prepare the image for neural encoding
    prepared_data, otsu_threshold, lower_threshold, upper_threshold = prepare_image_for_neural_encoding(
        input_image_path, output_edge_path
    )

    print(f"Otsu Threshold: {otsu_threshold}")
    print(f"Canny Thresholds -> Lower: {lower_threshold}, Upper: {upper_threshold}")
    print(f"Prepared Data (Flattened) for Neural Encoding: {prepared_data[:10]}...")  # Show a small part of the data 

