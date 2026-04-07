try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required but not installed. Run 'pip install pillow'."
    ) from exc

from pathlib import Path


def convert_to_rgb_png(input_path, output_path=None):
    """
    Reads any image type, converts it to RGB color space, and saves as PNG.

    Args:
        input_path: Path to the input image (any format Pillow supports).
        output_path: Optional output path. If None, saves alongside input with .png extension.

    Returns:
        Path to the saved PNG file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".png")
    else:
        output_path = Path(output_path)

    with Image.open(input_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(output_path, format="PNG")

    return output_path


def resize_proportional(input_path, output_path=None, resample_mode=Image.LANCZOS):
    """
    Proportionally resizes any image based on its longer side:
      - Below 1500 px  → upscale so longer side = 2000 px
      - 1500-3000 px   → no resize, just convert to RGB PNG
      - Above 3000 px  → downscale so longer side = 3000 px

    Args:
        input_path:    Path to the input image (any format).
        output_path:   Optional output path. If None, saves alongside input with _resized.png suffix.
        resample_mode: Pillow resampling filter (default LANCZOS).

    Returns:
        Tuple (output_path, new_width, new_height).
    """
    MIN_SIDE = 1500         # below this → upscale
    MAX_SIDE = 3000         # above this → downscale
    TARGET_UPSCALE = 2000   # target longer side when upscaling small images
    TARGET_DOWNSCALE = 3000 # target longer side when downscaling large images

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_resized.png")
    else:
        output_path = Path(output_path)

    with Image.open(input_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        longer_side = max(orig_w, orig_h)

        if longer_side < MIN_SIDE:
            # Too small → upscale so longer side reaches 2000
            scale = TARGET_UPSCALE / longer_side
        elif longer_side > MAX_SIDE:
            # Too large → downscale so longer side fits at 3000
            scale = TARGET_DOWNSCALE / longer_side
        else:
            # 1500–3000 px → no resize
            scale = 1.0

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        if scale != 1.0:
            img = img.resize((new_w, new_h), resample=resample_mode)
        img.save(output_path, format="PNG")

    return output_path, new_w, new_h


def enlarge_image(input_path, scale_input, output_path, resample_mode=Image.BILINEAR):
    """
    Enlarges an image by a float scale factor.
    scale_input: str or float (e.g., '1.5x', '2X', 3.0)
    """
    # 1. Clean the scale input
    if isinstance(scale_input, str):
        scale_str = scale_input.lower().replace("x", "")
        scale = float(scale_str)
    else:
        scale = float(scale_input)

    if scale <= 1.0:
        print(f"Warning: Scale {scale} is not enlarging the image.")

    # 2. Open image
    with Image.open(input_path) as img:
        # Convert unsupported modes for PNG output, e.g., CMYK -> RGB.
        if img.mode not in ("RGB", "RGBA", "L", "CMYK"):
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 3. Resize using simple interpolation
        enlarged_img = img.resize((new_w, new_h), resample=resample_mode)

        # 4. Save
        enlarged_img.save(output_path)
        return new_w, new_h


def main():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir.resolve()}")
        return

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    input_files = [
        p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_ext
    ]

    if not input_files:
        print(f"No image files found in: {input_dir.resolve()}")
        return

    scales_to_test = ["1.5x", "2.5x", "3.0x"]

    for image_path in input_files:
        for scale in scales_to_test:
            scale_tag = scale.lower().replace(".", "_")
            out_name = f"{image_path.stem}_enlarged_{scale_tag}.png"
            out_path = output_dir / out_name

            w, h = enlarge_image(image_path, scale, out_path)
            print(f"{image_path.name} -> {out_path} ({w}x{h})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to process images: {exc}")
