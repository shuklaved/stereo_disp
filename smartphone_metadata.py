# import exifread

# with open('/home/vedant/Documents/Projects_Flam/stereo_disp/data/IMG_20250424_150231207_HDR.jpg', "rb") as f:
#     tags = exifread.process_file(f)

# # Print relevant metadata
# print("Focal Length:", tags.get("EXIF FocalLength_"))
# print("Camera Make :", tags.get("Image Make"))
# print("Camera Model:", tags.get("Image Model"))

from PIL import Image
import exifread

# Load image
image_path = '/home/vedant/Documents/Projects_Flam/stereo_disp/data/IMG_20250424_150231207_HDR.jpg'

# Method 1: using Pillow
img = Image.open(image_path)
exif_data = img._getexif()

if exif_data:
    focal_length_mm = exif_data.get(37386)  # Tag 37386 = FocalLength
    if focal_length_mm:
        print(f"Focal length (mm): {float(focal_length_mm)} mm")

# Method 2: using ExifRead (better)
with open(image_path, 'rb') as f:
    tags = exifread.process_file(f)
    focal_length = tags.get('EXIF FocalLength')
    if focal_length:
        print(f"Focal length (mm): {float(focal_length_mm)} mm")

