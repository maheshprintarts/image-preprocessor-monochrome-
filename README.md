# image-preprocessor-monochrome

A small Python utility for preprocessing images before monochrome / single-color
pipelines. Built on top of [Pillow](https://python-pillow.org/), it standardises
arbitrary input formats into clean RGB PNGs and proportionally resizes them into
a predictable working range.

The whole tool lives in a single file:
[`SingleColor_pre_pro/single_color_pre_process.py`](SingleColor_pre_pro/single_color_pre_process.py).

## Requirements

- Python 3.8+
- Pillow

```bash
pip install pillow
```

## Features

The script exposes four building blocks: a format normaliser, a smart resizer,
a free-form enlarger, and a batch `main()` driver.

### 1. `convert_to_rgb_png(input_path, output_path=None)`

Reads an image in any format Pillow supports (JPG, PNG, BMP, WEBP, TIFF, ...),
converts the color space to **RGB**, and writes a PNG.

- Drops alpha and exotic modes (CMYK, P, LA, ...) into a clean RGB.
- If `output_path` is omitted, the file is saved next to the source with the
  `.png` extension.
- Returns the `Path` to the saved PNG.

```python
from SingleColor_pre_pro.single_color_pre_process import convert_to_rgb_png

convert_to_rgb_png("photo.tif")                # -> photo.png
convert_to_rgb_png("photo.jpg", "out/clean.png")
```

### 2. `resize_proportional(input_path, output_path=None, resample_mode=Image.LANCZOS)`

Proportionally resizes an image based on its **longer side**, keeping the
aspect ratio intact. The rules:

| Longer side of input | Action                                              |
| -------------------- | --------------------------------------------------- |
| `< 1500 px`          | **Upscale** so longer side becomes `2000 px`        |
| `1500 - 3000 px`     | **No resize** — only converted to RGB PNG           |
| `> 3000 px`          | **Downscale** so longer side becomes `3000 px`      |

This guarantees that every output image lands in the working range
**1500 - 3000 px** on its longer side, which is the sweet spot for the
downstream monochrome pipeline.

- Default resampling is `Image.LANCZOS` (high quality for both up- and downscale).
- Always converts to RGB and saves as PNG.
- If `output_path` is omitted, the file is saved as `<stem>_resized.png`
  next to the source.
- Returns `(output_path, new_width, new_height)`.

```python
from SingleColor_pre_pro.single_color_pre_process import resize_proportional

out, w, h = resize_proportional("input/test.jpg")
print(out, w, h)
```

### 3. `enlarge_image(input_path, scale_input, output_path, resample_mode=Image.BILINEAR)`

Enlarges an image by a free-form scale factor when you want to bypass the
size rules above and apply a fixed multiplier instead.

- `scale_input` accepts both strings and floats: `"1.5x"`, `"2X"`, `3.0`, ...
- Warns (via `print`) if the scale is `<= 1.0` (i.e. not actually enlarging).
- Preserves common modes (`RGB`, `RGBA`, `L`, `CMYK`); other modes are coerced
  to `RGB`.
- Default resampling is `Image.BILINEAR` (faster, lighter than LANCZOS — useful
  when you only need a quick preview).
- Returns `(new_width, new_height)`.

```python
from SingleColor_pre_pro.single_color_pre_process import enlarge_image

enlarge_image("input/test.jpg", "2.5x", "output/test_2_5x.png")
enlarge_image("input/test.jpg", 3.0,    "output/test_3x.png")
```

### 4. `main()` — batch enlarger

Running the script directly processes every image inside the local `input/`
folder and writes enlarged copies into `output/`:

- Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.
- Each image is enlarged at three preset scales: `1.5x`, `2.5x`, `3.0x`.
- Output filenames follow the pattern `<stem>_enlarged_<scale_tag>.png`,
  e.g. `photo_enlarged_2_5x.png`.
- Errors are caught at the top level and reported as
  `Failed to process images: <error>`.

```bash
mkdir -p input
cp some_photos/*.jpg input/
python SingleColor_pre_pro/single_color_pre_process.py
ls output/
```

## Quick reference

| Need                                            | Use                          |
| ----------------------------------------------- | ---------------------------- |
| Convert any image into a clean RGB PNG          | `convert_to_rgb_png`         |
| Get an image into the 1500 - 3000 px work range | `resize_proportional`        |
| Apply a fixed `Nx` scale (e.g. 2.5x)            | `enlarge_image`              |
| Batch-enlarge a folder of images at 1.5/2.5/3x  | run the script as `__main__` |
