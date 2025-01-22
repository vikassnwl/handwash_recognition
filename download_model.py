import os
import gdown

# Google Drive file ID
file_id = "1cwbLBiDToAJimYm83rnRNHuinK9PypO1"  # Replace with your Google Drive file ID

# Create the model directory if it doesn't exist
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Define the output file path
output_path = os.path.join(model_dir, "downloaded_model.keras")  # Change file name/extension if needed

# Download the file
gdown.download(url, output_path, quiet=False)

print(f"Model downloaded and saved to {output_path}")
