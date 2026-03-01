import cv2
import sys
import numpy as np

# --- Configuration ---

# 1. Your image file
image_path = 'IA_Section.png'

# 2. CONTROL SENSITIVITY - NOISE CLEANUP
#    Controls how big the black holes are to fill.
#    - Try 3, 5, 7, etc. (Bigger = fills larger holes)
#    - Set to 0 or 1 to disable hole-filling.
CLEANUP_STRENGTH = 9 # (This is the kernel_size)

# 3. CONTROL SENSITIVITY - BRIGHTNESS THRESHOLD
#    Controls what is "light" vs "dark".
#    - Set to 'None' to let the script decide AUTOMATICALLY (Recommended).
#    - Set to a number (0-255) to FORCE a cutoff.
#      (e.g., 128 means any pixel darker than 128 is BLACK)
MANUAL_THRESHOLD = None

# --- End of Configuration ---


# 1. Load the image
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. Check if the image was loaded successfully
if gray_image is None:
    print(f"Error: Could not load image from path: {image_path}")
    sys.exit()

# 3. Apply thresholding (Automatic or Manual)
print("--- Applying Threshold ---")
if MANUAL_THRESHOLD is None:
    # Use Otsu's method to find the best threshold automatically
    optimal_threshold_value, binary_light_is_white = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Using Automatic (Otsu) Threshold: {optimal_threshold_value:.0f}")
else:
    # Use the manually set threshold value
    optimal_threshold_value = MANUAL_THRESHOLD
    _, binary_light_is_white = cv2.threshold(
        gray_image, optimal_threshold_value, 255, cv2.THRESH_BINARY
    )
    print(f"Using Manual Threshold: {optimal_threshold_value}")

# Save the original binary image for comparison
output_original_binary_path = 'output_01_original_binary.png'
cv2.imwrite(output_original_binary_path, binary_light_is_white)

# 4. Clean the image (if strength is > 1)
if CLEANUP_STRENGTH > 1:
    print(f"Applying cleanup with strength: {CLEANUP_STRENGTH}")
    # Define the kernel
    kernel = np.ones((CLEANUP_STRENGTH, CLEANUP_STRENGTH), np.uint8)

    # Apply "Morphological Closing"
    cleaned_image = cv2.morphologyEx(binary_light_is_white, cv2.MORPH_CLOSE, kernel)

    # The image to be counted is the cleaned one
    image_to_count = cleaned_image
    output_final_path = 'output_02_cleaned_final.png'
else:
    print("Cleanup skipped (strength was 0 or 1).")
    # The image to be counted is the original binary one
    image_to_count = binary_light_is_white
    output_final_path = 'output_02_final_no_cleanup.png'

# 5. Count the pixels (based on the final processed image)
try:
    total_pixels = gray_image.size

    # Count non-zero (white) pixels in the final image
    light_pixels = cv2.countNonZero(image_to_count)
    dark_pixels = total_pixels - light_pixels

    light_percentage = (light_pixels / total_pixels) * 100
    dark_percentage = (dark_pixels / total_pixels) * 100

    # 6. Print the results
    print(f"\n--- Image Analysis Complete ---")
    print(f"Source Image: {image_path}")

    print("\n--- Pixel Counts (Based on Final Image) ---")
    print(f"Total Pixels: {total_pixels:,}")
    print(f"Light PixE: {light_pixels:,} ({light_percentage:.2f}%)")
    print(f"Dark Pixels:  {dark_pixels:,} ({dark_percentage:.2f}%)")

    # 7. Save the new images
    output_dark_path = 'output_03_dark_pixels_as_white.png'

    # Save the NEW final image (either cleaned or not)
    cv2.imwrite(output_final_path, image_to_count)

    # We also still need the inverted image
    # Note: We re-threshold here to ensure it respects the manual threshold
    if MANUAL_THRESHOLD is None:
        _, binary_dark_is_white = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary_dark_is_white = cv2.threshold(
            gray_image, MANUAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

    cv2.imwrite(output_dark_path, binary_dark_is_white)

    print("\n--- Output Files Saved ---")
    print(f"1. {output_original_binary_path} (Raw binary, before cleanup)")
    print(f"2. {output_final_path} (FINAL image used for counting)")
    print(f"3. {output_dark_path} (Inverted final image)")

except cv2.error as e:
    print(f"An OpenCV error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")