try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required but not installed. Run 'pip install opencv-python'."
    ) from exc

from pathlib import Path


def canny_edge(input_path, output_path=None, threshold1=100, threshold2=200,
               blur_kernel=5, invert=False):
    """
    Reads any image, applies Canny edge detection, and saves the result as PNG.

    Args:
        input_path:  Path to the input image (any format OpenCV supports).
        output_path: Optional output path. If None, saves alongside input with _canny.png suffix.
        threshold1:  Lower threshold for the hysteresis (default 100).
        threshold2:  Upper threshold for the hysteresis (default 200).
        blur_kernel: Gaussian blur kernel size before edge detection (odd number, default 5).
                     Higher value = smoother edges, less noise.
        invert:      If True, white edges on black background → black edges on white background.

    Returns:
        Path to the saved PNG file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_canny.png")
    else:
        output_path = Path(output_path)

    # Read image (any format)
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Convert to grayscale (Canny requires single channel)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # already grayscale

    # Gaussian blur to reduce noise before edge detection
    if blur_kernel > 1:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1  # must be odd
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Invert: black edges on white background
    if invert:
        edges = cv2.bitwise_not(edges)

    cv2.imwrite(str(output_path), edges)
    return output_path


def process_folder(input_dir="input", output_dir="output", threshold1=100, threshold2=200,
                   blur_kernel=5, invert=False):
    """
    Batch-processes all images in input_dir, saves Canny outputs to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir.resolve()}")
        return

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    input_files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in allowed_ext
    ]

    if not input_files:
        print(f"No image files found in: {input_dir.resolve()}")
        return

    for image_path in input_files:
        out_path = output_dir / (image_path.stem + "_canny.png")
        result = canny_edge(image_path, out_path, threshold1, threshold2, blur_kernel, invert)
        print(f"{image_path.name} -> {result}")


if __name__ == "__main__":
    process_folder(
        input_dir="input",
        output_dir="output",
        threshold1=100,   # lower threshold  (lower = more edges detected)
        threshold2=200,   # upper threshold  (higher = only strong edges)
        blur_kernel=5,    # noise reduction before detection
        invert=False,     # True = black edges on white background
    )
