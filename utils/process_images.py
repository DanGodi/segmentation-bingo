from PIL import Image
import os
from pathlib import Path

from sklearn import base

def process_images(input_folder, output_folder, scale=1.0):
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect and sort only image files, then enumerate to rename sequentially
    image_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    )

    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Downscale
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Save as high-quality JPEG with sequential name: image_1.jpg, image_2.jpg, ...
        out_name = f"image_{idx}.jpg"
        out_path = os.path.join(output_folder, out_name)
        img_resized.convert("RGB").save(out_path, "JPEG", quality=90)

        print("Saved:", out_path)

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent
    input_folder = REPO_ROOT / "sat_images"
    output_folder = REPO_ROOT / "converted_sat_images"
    process_images(input_folder, output_folder)


