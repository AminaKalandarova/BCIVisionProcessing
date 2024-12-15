import numpy as np
import cv2
import pandas as pd

# Step 1: Load flattened data from CSV
flattened_data = pd.read_csv('flattened_data.csv', header=None).values.flatten()

# Step 2: Check the size of the flattened data
data_size = flattened_data.size
print(f"Flattened data size: {data_size}")

# Step 3: Calculate the next perfect square size
next_square_size = int(np.ceil(np.sqrt(data_size)))  # Next perfect square size

# Step 4: Pad the data if it's not a perfect square
padded_data = np.pad(flattened_data, (0, next_square_size**2 - data_size), 'constant', constant_values=0)

# Step 5: Reshape into a square grid (next perfect square)
electrode_grid = padded_data.reshape(next_square_size, next_square_size)

# Step 6: Map 1s to stimulated (255) and 0s to non-stimulated (0)
stimulated_grid = electrode_grid * 255  # Convert 1s to 255 (stimulated), 0s to 0 (non-stimulated)

# Step 7: Convert the stimulated grid to uint8 for image display
stimulated_grid = stimulated_grid.astype(np.uint8)

# Step 8: Display the simulated neural stimulation
cv2.imshow("Prosthetic Neural Stimulation", stimulated_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 9: Optionally, save the result as an image
cv2.imwrite('prosthetic_neural_stimulation.jpg', stimulated_grid)
print("Prosthetic Neural Stimulation image saved as 'prosthetic_neural_stimulation.jpg'")


