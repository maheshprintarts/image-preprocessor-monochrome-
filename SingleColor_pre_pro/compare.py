import sys
import re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import cv2
except ImportError as exc:
    raise ImportError("Run 'pip install opencv-python'") from exc

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("Run 'pip install numpy'") from exc

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc:
    raise ImportError("Run 'pip install scikit-image'") from exc

from pathlib import Path


ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Minimum contour area (pixels) to count as a changed segment
MIN_SEGMENT_AREA = 200


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return round(10 * np.log10(255.0 ** 2 / mse), 2)


def _ssim_score(a_gray, b_gray):
    score, diff = ssim(a_gray, b_gray, full=True)
    diff = (diff * 255).astype(np.uint8)
    return round(float(score), 4), diff


def _change_label(score):
    if score >= 0.97: return "Almost identical"
    if score >= 0.90: return "Minor change"
    if score >= 0.75: return "Moderate change"
    if score >= 0.50: return "Major change"
    return "Heavily changed"


# ---------------------------------------------------------------------------
# Segment / region detection
# ---------------------------------------------------------------------------

def _find_changed_segments(in_bgr, out_bgr, threshold=30):
    """
    Finds changed regions between two same-size BGR images.

    Returns:
        segments   : list of dicts per region
        mask       : binary mask of all changed pixels
        abs_diff   : per-pixel absolute difference (grayscale)
    """
    abs_diff = cv2.absdiff(in_bgr, out_bgr)
    diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)

    # Threshold → binary change mask
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological close: merge nearby changed pixels into solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of changed regions
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_pixels = in_bgr.shape[0] * in_bgr.shape[1]
    changed_pixels = int(np.count_nonzero(mask))

    segments = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_SEGMENT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Crop region from both images
        roi_in  = in_bgr[y:y+h, x:x+w]
        roi_out = out_bgr[y:y+h, x:x+w]
        roi_diff = abs_diff[y:y+h, x:x+w]

        # Per-channel mean color shift (B, G, R)
        ch_shift = np.mean(roi_diff, axis=(0, 1))  # [B, G, R]

        # Mean brightness before / after
        bright_in  = float(np.mean(cv2.cvtColor(roi_in,  cv2.COLOR_BGR2GRAY)))
        bright_out = float(np.mean(cv2.cvtColor(roi_out, cv2.COLOR_BGR2GRAY)))

        segments.append({
            "id":           i + 1,
            "x": x, "y": y, "w": w, "h": h,
            "area_px":      int(area),
            "area_pct":     round(area / total_pixels * 100, 2),
            "mean_diff":    round(float(np.mean(roi_diff)), 2),
            "max_diff":     round(float(np.max(roi_diff)), 2),
            "shift_B":      round(float(ch_shift[0]), 1),
            "shift_G":      round(float(ch_shift[1]), 1),
            "shift_R":      round(float(ch_shift[2]), 1),
            "bright_in":    round(bright_in, 1),
            "bright_out":   round(bright_out, 1),
            "bright_delta": round(bright_out - bright_in, 1),
        })

    # Sort largest segment first
    segments.sort(key=lambda s: s["area_px"], reverse=True)

    return segments, mask, diff_gray, changed_pixels, total_pixels


# ---------------------------------------------------------------------------
# Visual output builder
# ---------------------------------------------------------------------------

def _build_visual(in_bgr, out_bgr, segments, mask, diff_gray, out_path):
    """
    Saves a 4-panel annotated image:
      Panel 1 - INPUT  with numbered segment boxes
      Panel 2 - OUTPUT with numbered segment boxes
      Panel 3 - Heatmap of pixel-level change intensity
      Panel 4 - Changed segments mask (white = changed)
    """
    TARGET_H = 400
    h0 = in_bgr.shape[0]

    def fit(img):
        scale = TARGET_H / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), TARGET_H),
                          interpolation=cv2.INTER_AREA), scale

    in_disp,  sc_in  = fit(in_bgr.copy())
    out_disp, sc_out = fit(out_bgr.copy())

    # Draw numbered boxes on input and output panels
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255),
              (255, 0, 255), (255, 255, 0), (0, 255, 255)]

    for seg in segments:
        col = colors[(seg["id"] - 1) % len(colors)]
        for disp, sc in [(in_disp, sc_in), (out_disp, sc_out)]:
            bx = int(seg["x"] * sc); by = int(seg["y"] * sc)
            bw = int(seg["w"] * sc); bh = int(seg["h"] * sc)
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), col, 2)
            cv2.putText(disp, str(seg["id"]), (bx + 4, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Heatmap panel
    heatmap_raw = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)
    heatmap_disp, _ = fit(heatmap)

    # Mask panel
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_disp, _ = fit(mask_3ch)

    # Draw segment outlines on mask
    for seg in segments:
        col = colors[(seg["id"] - 1) % len(colors)]
        sc = TARGET_H / h0
        bx = int(seg["x"] * sc); by = int(seg["y"] * sc)
        bw = int(seg["w"] * sc); bh = int(seg["h"] * sc)
        cv2.rectangle(mask_disp, (bx, by), (bx + bw, by + bh), col, 2)
        cv2.putText(mask_disp, str(seg["id"]), (bx + 4, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Pad all panels to same width before hstack
    max_w = max(p.shape[1] for p in [in_disp, out_disp, heatmap_disp, mask_disp])

    def pad_w(img):
        pad = max_w - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    panels = [pad_w(p) for p in [in_disp, out_disp, heatmap_disp, mask_disp]]
    row = np.hstack(panels)

    # Label strip at top
    label_h = 28
    label_bar = np.zeros((label_h, row.shape[1], 3), dtype=np.uint8)
    labels = ["INPUT (segments)", "OUTPUT (segments)", "HEATMAP (intensity)", "CHANGE MASK"]
    for i, lbl in enumerate(labels):
        cv2.putText(label_bar, lbl,
                    (i * max_w + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    final = np.vstack([label_bar, row])
    cv2.imwrite(str(out_path), final)


# ---------------------------------------------------------------------------
# Public: compare one pair
# ---------------------------------------------------------------------------

def compare_images(input_path, output_path, diff_path=None, threshold=30):
    """
    Compares input vs output image.
    Measures SSIM, PSNR, pixel-change %, and detects changed segments.

    Args:
        input_path:  Original image.
        output_path: Processed image.
        diff_path:   Optional path to save the 4-panel visual PNG.
        threshold:   Pixel difference value (0-255) to count as changed (default 30).

    Returns:
        dict with full metrics + 'segments' list.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    in_bgr  = _load_bgr(input_path)
    out_bgr = _load_bgr(output_path)

    # Normalise to same size for comparison
    if in_bgr.shape != out_bgr.shape:
        out_bgr = cv2.resize(out_bgr, (in_bgr.shape[1], in_bgr.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)

    in_gray  = cv2.cvtColor(in_bgr,  cv2.COLOR_BGR2GRAY)
    out_gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)

    ssim_val, _   = _ssim_score(in_gray, out_gray)
    psnr_val      = _psnr(in_bgr, out_bgr)

    segments, mask, diff_gray, changed_px, total_px = \
        _find_changed_segments(in_bgr, out_bgr, threshold)

    changed_pct = round(changed_px / total_px * 100, 2)

    if diff_path is not None:
        diff_path = Path(diff_path)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        _build_visual(in_bgr, out_bgr, segments, mask, diff_gray, diff_path)

    ih, iw = _load_bgr(input_path).shape[:2]
    oh, ow = _load_bgr(output_path).shape[:2]

    return {
        "ssim":         ssim_val,
        "psnr":         psnr_val,
        "change":       _change_label(ssim_val),
        "changed_pct":  changed_pct,
        "segments":     segments,
        "input_size":   f"{iw}x{ih}",
        "output_size":  f"{ow}x{oh}",
    }


# ---------------------------------------------------------------------------
# Public: batch folder compare
# ---------------------------------------------------------------------------

def compare_folders(input_dir="input", output_dir="output", diff_dir="diff", threshold=30):
    """
    Matches input/ vs output/ by stem name, compares each pair,
    prints a full segment report, and saves 4-panel visual diffs.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    diff_dir   = Path(diff_dir)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir.resolve()}"); return
    if not output_dir.exists():
        print(f"Output folder not found: {output_dir.resolve()}"); return

    input_files = [p for p in sorted(input_dir.iterdir())
                   if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    output_files = [p for p in sorted(output_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT]

    # Match by stem prefix
    output_lookup = {}
    for op in output_files:
        for ip in input_files:
            if re.match(r'^' + re.escape(ip.stem) + r'([^a-zA-Z0-9]|$)', op.stem):
                output_lookup.setdefault(ip.stem, op)

    sep     = "=" * 110
    summary = []   # collect results for final summary table

    for ip in input_files:
        op = output_lookup.get(ip.stem)
        if op is None:
            print(f"\n{sep}")
            print(f"  {ip.name}  ->  No matching output found")
            summary.append({"name": ip.name, "changed_pct": None, "change": "No output"})
            continue

        diff_path = diff_dir / (ip.stem + "_compare.png")

        try:
            r = compare_images(ip, op, diff_path, threshold)
        except Exception as exc:
            print(f"\n{sep}\n  {ip.name}  ->  ERROR: {exc}")
            summary.append({"name": ip.name, "changed_pct": None, "change": f"ERROR: {exc}"})
            continue

        sim_pct   = round(r["ssim"] * 100, 2)          # SSIM-based similarity %
        pixel_sim = round(100 - r["changed_pct"], 2)   # pixel-level similarity %

        print(f"\n{sep}")
        print(f"  FILE       : {ip.name}  ->  {op.name}")
        print(f"  SIZE       : {r['input_size']}  ->  {r['output_size']}")
        print(f"  SIMILARITY : {sim_pct}%  (structural)   |   {pixel_sim}%  (pixel match)   ->  {r['change']}")
        print(f"  PSNR       : {r['psnr']} dB")
        print(f"  SEGMENTS   : {len(r['segments'])} changed region(s) detected")
        print(f"  DIFF IMG   : {diff_path}")

        if r["segments"]:
            print()
            print(f"  {'Seg':>3}  {'Area%':>6}  {'AreaPx':>8}  {'MeanDiff':>9}  "
                  f"{'MaxDiff':>8}  {'BrightD':>8}  {'Shift B/G/R':>14}  Location (x,y,w,h)")
            print("  " + "-" * 95)
            for s in r["segments"]:
                print(
                    f"  {s['id']:>3}  "
                    f"{s['area_pct']:>6.2f}%  "
                    f"{s['area_px']:>8,}  "
                    f"{s['mean_diff']:>9.1f}  "
                    f"{s['max_diff']:>8.1f}  "
                    f"{s['bright_delta']:>+8.1f}  "
                    f"{s['shift_B']:>4.0f}/{s['shift_G']:>4.0f}/{s['shift_R']:>4.0f}  "
                    f"  ({s['x']},{s['y']},{s['w']},{s['h']})"
                )

        summary.append({
            "name":      ip.name,
            "output":    op.name,
            "ssim":      r["ssim"],
            "psnr":      r["psnr"],
            "sim_pct":   sim_pct,
            "pixel_sim": pixel_sim,
            "change":    r["change"],
        })

    # ------------------------------------------------------------------
    # FINAL SUMMARY TABLE
    # ------------------------------------------------------------------
    print(f"\n\n{'#' * 110}")
    print(f"  FINAL SUMMARY  -  Image Similarity Report")
    print(f"{'#' * 110}")
    print(f"  {'#':>3}  {'Input File':<35}  {'Similarity':>11}  {'Pixel Match':>12}  {'PSNR (dB)':>10}  Verdict")
    print("  " + "-" * 105)

    for i, s in enumerate(summary, 1):
        if s.get("sim_pct") is None:
            print(f"  {i:>3}  {s['name']:<35}  {'---':>11}  {'---':>12}  {'---':>10}  {s['change']}")
        else:
            bar_len = int(s["sim_pct"] / 2)          # scale 100% -> 50 chars
            bar     = "[" + "#" * bar_len + "." * (50 - bar_len) + "]"
            print(f"  {i:>3}  {s['name']:<35}  {s['sim_pct']:>10.2f}%  {s['pixel_sim']:>11.2f}%  {s['psnr']:>10}  {s['change']}")
            print(f"       {bar}  {s['sim_pct']:.1f}% similar")

    print(f"\n{'#' * 110}")
    print(f"  Diff images saved to : {diff_dir.resolve()}")
    print(f"{'#' * 110}\n")
    print("COLUMN GUIDE")
    print("  Similarity  - structural similarity % (SSIM x 100) — best overall measure")
    print("  Pixel Match - % of pixels that are the same (100 - changed pixels %)")
    print("  PSNR        - inf=identical  40+dB=high quality  30-40=OK  <30=noticeable loss")
    print("  Verdict     - 97%+=almost identical  90%+=minor  75%+=moderate  <50%=heavily different")
    print()


# ---------------------------------------------------------------------------
# Background-subtraction based segment diff  (new technique)
# ---------------------------------------------------------------------------

def _estimate_background(bgr, border=40):
    """
    Estimates background color as the median of a thin border strip around the image.
    Works for white, grey, black, or any solid-ish background.
    """
    h, w = bgr.shape[:2]
    b = border
    strips = np.vstack([
        bgr[:b, :].reshape(-1, 3),          # top
        bgr[-b:, :].reshape(-1, 3),         # bottom
        bgr[:, :b].reshape(-1, 3),          # left
        bgr[:, -b:].reshape(-1, 3),         # right
    ])
    return np.median(strips, axis=0).astype(np.uint8)


def _content_mask(bgr, bg_color, tolerance=25):
    """
    Returns a binary mask: white = content (different from background),
    black = background.
    Uses per-pixel max-channel distance from background color.
    """
    diff  = np.abs(bgr.astype(np.int16) - bg_color.astype(np.int16))
    dist  = np.max(diff, axis=2).astype(np.uint8)
    _, mask = cv2.threshold(dist, tolerance, 255, cv2.THRESH_BINARY)

    # Clean noise: open (remove tiny specs) then close (fill small holes)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def _mask_to_regions(mask, min_area=400):
    """
    Finds connected components in a binary mask.
    Returns list of region dicts sorted by area descending.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    regions = []
    for i in range(1, num_labels):           # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        w  = int(stats[i, cv2.CC_STAT_WIDTH])
        h  = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        regions.append({"id": i, "x": x, "y": y, "w": w, "h": h,
                         "area": area, "cx": cx, "cy": cy})
    regions.sort(key=lambda r: r["area"], reverse=True)
    for idx, r in enumerate(regions):
        r["id"] = idx + 1
    return regions


def _build_segment_diff_visual(in_bgr, out_bgr, missing, added, kept,
                                in_mask, out_mask, out_path):
    """
    Saves a 4-panel visual:
      Panel 1 – INPUT  with coloured region boxes
      Panel 2 – OUTPUT with coloured region boxes
      Panel 3 – INPUT content mask  (white = content)
      Panel 4 – OUTPUT content mask (white = content)
    Colours:
      Green  = content present in both (kept)
      Red    = MISSING from output
      Yellow = NEW in output (not in input)
    """
    TARGET_H = 500

    def fit(img):
        sc = TARGET_H / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * sc), TARGET_H),
                          interpolation=cv2.INTER_AREA), sc

    in_disp,  sc_in  = fit(in_bgr.copy())
    out_disp, sc_out = fit(out_bgr.copy())
    in_mask_disp,  _ = fit(cv2.cvtColor(in_mask,  cv2.COLOR_GRAY2BGR))
    out_mask_disp, _ = fit(cv2.cvtColor(out_mask, cv2.COLOR_GRAY2BGR))

    def draw_box(img, reg, sc, color, label):
        x = int(reg["x"] * sc); y = int(reg["y"] * sc)
        w = int(reg["w"] * sc); h = int(reg["h"] * sc)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x + 3, max(y + 16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    for r in kept:
        draw_box(in_disp,  r, sc_in,  (0, 200, 0),   f"#{r['id']}")
    for r in missing:
        draw_box(in_disp,  r, sc_in,  (0, 0, 255),   f"MISS#{r['id']}")
    for r in kept:
        draw_box(out_disp, r, sc_out, (0, 200, 0),   f"#{r['id']}")
    for r in added:
        draw_box(out_disp, r, sc_out, (0, 200, 255), f"NEW#{r['id']}")

    pad = max(p.shape[1] for p in [in_disp, out_disp, in_mask_disp, out_mask_disp])

    def pad_w(img):
        p = pad - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, p, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    row = np.hstack([pad_w(in_disp), pad_w(out_disp),
                     pad_w(in_mask_disp), pad_w(out_mask_disp)])

    labels_bar = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    for i, lbl in enumerate(["INPUT", "OUTPUT", "INPUT MASK", "OUTPUT MASK"]):
        cv2.putText(labels_bar, lbl, (i * pad + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    legend = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend,
                "GREEN=present in both   RED=MISSING from output   YELLOW=NEW in output",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imwrite(str(out_path), np.vstack([labels_bar, row, legend]))


# ---------------------------------------------------------------------------
# Public: segment element diff for one pair
# ---------------------------------------------------------------------------

def segment_diff(input_path, output_path, visual_path=None,
                 min_area=400, tolerance=25):
    """
    Detects content regions in input and output via background subtraction,
    then reports which input regions are MISSING in output and which are NEW.

    Technique:
      1. Estimate background from image border (works for any solid background)
      2. Build content mask = pixels significantly different from background
      3. Find connected components in each mask → content regions
      4. missing = regions in input mask  NOT covered in output mask
      5. added   = regions in output mask NOT covered in input mask
      6. kept    = regions present in both

    Args:
        input_path:  Original image.
        output_path: Processed image.
        visual_path: Optional path to save the 4-panel visual PNG.
        min_area:    Minimum pixel area to count as a region (default 400).
        tolerance:   How different a pixel must be from background to count
                     as content (0-255, default 25).

    Returns:
        dict with region lists and counts.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    in_bgr  = _load_bgr(input_path)
    out_bgr = _load_bgr(output_path)

    if in_bgr.shape != out_bgr.shape:
        out_bgr = cv2.resize(out_bgr, (in_bgr.shape[1], in_bgr.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)

    # Step 1 — background colours
    in_bg  = _estimate_background(in_bgr)
    out_bg = _estimate_background(out_bgr)

    # Step 2 — content masks
    in_mask  = _content_mask(in_bgr,  in_bg,  tolerance)
    out_mask = _content_mask(out_bgr, out_bg, tolerance)

    # Step 3 — content regions from input mask
    in_regions = _mask_to_regions(in_mask, min_area)

    # Step 4 — classify each input region
    missing, kept = [], []
    for reg in in_regions:
        x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
        roi_out_mask = out_mask[y:y+h, x:x+w]
        coverage = np.count_nonzero(roi_out_mask) / (w * h)
        if coverage < 0.15:          # less than 15% of region covered in output → missing
            missing.append(reg)
        else:
            kept.append(reg)

    # Step 5 — find NEW regions in output not covered by input mask
    out_regions  = _mask_to_regions(out_mask, min_area)
    added = []
    for reg in out_regions:
        x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
        roi_in_mask = in_mask[y:y+h, x:x+w]
        coverage = np.count_nonzero(roi_in_mask) / (w * h)
        if coverage < 0.15:
            added.append(reg)

    if visual_path is not None:
        visual_path = Path(visual_path)
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        _build_segment_diff_visual(in_bgr, out_bgr, missing, added, kept,
                                   in_mask, out_mask, visual_path)

    return {
        "in_total":  len(in_regions),
        "out_total": len(out_regions),
        "kept":      len(kept),
        "missing":   missing,
        "added":     added,
        "in_bg":     in_bg.tolist(),
        "out_bg":    out_bg.tolist(),
    }


# ---------------------------------------------------------------------------
# Public: batch segment diff across folders
# ---------------------------------------------------------------------------

def segment_diff_folders(input_dir="input", output_dir="output", visual_dir="diff_segments",
                         min_area=400, tolerance=25):
    """
    Runs segment_diff() on all matched input/output pairs and prints a full report.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    visual_dir = Path(visual_dir)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir.resolve()}"); return
    if not output_dir.exists():
        print(f"Output folder not found: {output_dir.resolve()}"); return

    input_files  = [p for p in sorted(input_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    output_files = [p for p in sorted(output_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT]

    output_lookup = {}
    for op in output_files:
        for ip in input_files:
            if re.match(r'^' + re.escape(ip.stem) + r'([^a-zA-Z0-9]|$)', op.stem):
                output_lookup.setdefault(ip.stem, op)

    sep      = "=" * 110
    summary  = []

    for ip in input_files:
        op = output_lookup.get(ip.stem)
        if op is None:
            print(f"\n{sep}")
            print(f"  {ip.name}  ->  No matching output found")
            summary.append({"name": ip.name, "result": None})
            continue

        visual_path = visual_dir / (ip.stem + "_segments.png")

        try:
            r = segment_diff(ip, op, visual_path, min_area, tolerance)
        except Exception as exc:
            print(f"\n{sep}\n  {ip.name}  ->  ERROR: {exc}")
            summary.append({"name": ip.name, "in_total": None})
            continue

        missing_pct = round(len(r["missing"]) / r["in_total"] * 100, 1) if r["in_total"] else 0
        added_pct   = round(len(r["added"])   / r["in_total"] * 100, 1) if r["in_total"] else 0

        in_bg  = r["in_bg"]
        out_bg = r["out_bg"]

        print(f"\n{sep}")
        print(f"  FILE        : {ip.name}  ->  {op.name}")
        print(f"  BACKGROUND  : Input  RGB({in_bg[2]},{in_bg[1]},{in_bg[0]})   "
              f"Output RGB({out_bg[2]},{out_bg[1]},{out_bg[0]})")
        print(f"  REGIONS     : Input={r['in_total']}   Output={r['out_total']}   "
              f"Kept={r['kept']}   Missing={len(r['missing'])}   New={len(r['added'])}")
        print(f"  MISSING     : {len(r['missing'])} region(s) from input are ABSENT in output  ({missing_pct}%)")
        print(f"  NEW         : {len(r['added'])} region(s) in output were NOT in input  ({added_pct}%)")
        print(f"  VISUAL      : {visual_path}")

        if r["missing"]:
            print()
            print(f"  --- MISSING regions (present in input, GONE from output) ---")
            print(f"  {'#':>3}  {'Area (px)':>10}  {'% of image':>11}  {'Location (x,y,w,h)'}")
            print("  " + "-" * 60)
            _tmp     = _load_bgr(ip)
            total_px = _tmp.shape[0] * _tmp.shape[1]
            for e in r["missing"]:
                pct = round(e["area"] / total_px * 100, 2)
                print(f"  {e['id']:>3}  {e['area']:>10,}  {pct:>10.2f}%  ({e['x']},{e['y']},{e['w']},{e['h']})")

        if r["added"]:
            print()
            print(f"  --- NEW regions (in output, NOT in input) ---")
            print(f"  {'#':>3}  {'Area (px)':>10}  {'Location (x,y,w,h)'}")
            print("  " + "-" * 50)
            for e in r["added"]:
                print(f"  {e['id']:>3}  {e['area']:>10,}  ({e['x']},{e['y']},{e['w']},{e['h']})")

        summary.append({
            "name":        ip.name,
            "in_total":    r["in_total"],
            "out_total":   r["out_total"],
            "kept":        r["kept"],
            "missing":     len(r["missing"]),
            "added":       len(r["added"]),
            "missing_pct": missing_pct,
        })

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n\n{'#' * 110}")
    print(f"  SEGMENT DIFF SUMMARY  -  Missing / Added Regions per Image")
    print(f"{'#' * 110}")
    print(f"  {'#':>3}  {'Input File':<35}  {'In':>5}  {'Out':>5}  {'Kept':>5}  "
          f"{'Missing':>8}  {'New':>5}  {'Missing %':>10}")
    print("  " + "-" * 95)

    for i, s in enumerate(summary, 1):
        if s.get("in_total") is None:
            print(f"  {i:>3}  {s['name']:<35}  {'---':>5}  {'---':>5}  {'---':>5}  {'---':>8}  {'---':>5}")
        else:
            total   = max(s["in_total"], 1)
            bar_ok  = int(s["kept"]    / total * 30)
            bar_mis = int(s["missing"] / total * 30)
            bar_add = max(0, 30 - bar_ok - bar_mis)
            bar = "[" + "=" * bar_ok + "X" * bar_mis + "+" * bar_add + "]"
            print(f"  {i:>3}  {s['name']:<35}  {s['in_total']:>5}  {s['out_total']:>5}  "
                  f"{s['kept']:>5}  {s['missing']:>8}  {s['added']:>5}  {s['missing_pct']:>9.1f}%")
            print(f"       {bar}  (= kept   X missing   + new in output)")

    print(f"\n{'#' * 110}")
    print(f"  Segment visuals saved to : {visual_dir.resolve()}")
    print(f"{'#' * 110}\n")
    print("COLUMN GUIDE")
    print("  In       - content regions detected in input (by background subtraction)")
    print("  Out      - content regions detected in output")
    print("  Kept     - input regions still present in output")
    print("  Missing  - input regions GONE from output  (red boxes in visual)")
    print("  New      - output regions not in input     (yellow boxes in visual)")
    print("  Missing% - percentage of input content lost in output")
    print()


# ---------------------------------------------------------------------------
# Canny edge-based similarity
# ---------------------------------------------------------------------------

def _canny_map(bgr, blur=5, t1=50, t2=150):
    """
    Returns a normalised Canny edge map for one image.
    Auto-threshold using Otsu if t1/t2 are 0.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        blur = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, t1, t2)
    return edges


def _edge_similarity(edges_a, edges_b):
    """
    Compares two binary edge maps and returns three scores:

    1. F1 (edge overlap)
       Treats edge pixels as positives.
       precision = matched / predicted,  recall = matched / actual
       F1 = harmonic mean of precision and recall
       → best measure: rewards finding the same edges, penalises false edges

    2. IoU (Intersection over Union of edge pixels)
       intersection / union of white pixels
       → strict: 1.0 only when edge maps are exactly equal

    3. SSIM on edge maps
       structural similarity of the edge image as a whole
    """
    a = edges_a.astype(bool)
    b = edges_b.astype(bool)

    intersection = np.count_nonzero(a & b)
    union        = np.count_nonzero(a | b)
    pred         = np.count_nonzero(b)
    actual       = np.count_nonzero(a)

    precision = intersection / pred   if pred   > 0 else 0.0
    recall    = intersection / actual if actual > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = intersection / union if union > 0 else 1.0  # both empty = identical

    ssim_val, _ = _ssim_score(edges_a, edges_b)
    ssim_val    = max(0.0, float(ssim_val))

    return round(f1 * 100, 2), round(iou * 100, 2), round(ssim_val * 100, 2)


def _canny_verdict(f1):
    if f1 >= 85: return "Structurally identical"
    if f1 >= 65: return "Very similar structure"
    if f1 >= 45: return "Moderate structural match"
    if f1 >= 25: return "Low structural match"
    return "Structurally different"


def canny_similarity(in_bgr, out_bgr, blur=5, t1=50, t2=150):
    """
    Computes Canny-edge-based similarity between two BGR images.
    Images must already be the same size.
    Returns dict: f1, iou, ssim, verdict
    """
    e_in  = _canny_map(in_bgr,  blur, t1, t2)
    e_out = _canny_map(out_bgr, blur, t1, t2)
    f1, iou, ssim_e = _edge_similarity(e_in, e_out)
    return {
        "f1":      f1,
        "iou":     iou,
        "ssim":    ssim_e,
        "verdict": _canny_verdict(f1),
        "e_in":    e_in,
        "e_out":   e_out,
    }


def final_similarity_report(input_dir="input", output_dir="output"):
    """
    Prints a clean, focused similarity percentage for every input/output pair.
    No segment detail — just the final answer: how similar are these two images?
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir.resolve()}"); return
    if not output_dir.exists():
        print(f"Output folder not found: {output_dir.resolve()}"); return

    input_files  = [p for p in sorted(input_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    output_files = [p for p in sorted(output_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT]

    output_lookup = {}
    for op in output_files:
        for ip in input_files:
            if re.match(r'^' + re.escape(ip.stem) + r'([^a-zA-Z0-9]|$)', op.stem):
                output_lookup.setdefault(ip.stem, op)

    results = []
    for ip in input_files:
        op = output_lookup.get(ip.stem)
        if op is None:
            results.append({"name": ip.name, "sim": None, "verdict": "No output found"})
            continue
        try:
            in_bgr  = _load_bgr(ip)
            out_bgr = _load_bgr(op)
            if in_bgr.shape != out_bgr.shape:
                out_bgr = cv2.resize(out_bgr, (in_bgr.shape[1], in_bgr.shape[0]),
                                     interpolation=cv2.INTER_LANCZOS4)

            # SSIM — overall structural similarity
            in_gray  = cv2.cvtColor(in_bgr,  cv2.COLOR_BGR2GRAY)
            out_gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
            score, _ = _ssim_score(in_gray, out_gray)
            ssim_pct = round(score * 100, 2)

            # Canny edge similarity — structure only, ignores color/brightness
            c = canny_similarity(in_bgr, out_bgr)

            # Combined score: SSIM 40% + Canny F1 40% + Canny IoU 20%
            combined = round(0.40 * ssim_pct + 0.40 * c["f1"] + 0.20 * c["iou"], 2)

            results.append({
                "name":     ip.name,
                "output":   op.name,
                "ssim":     ssim_pct,
                "edge_f1":  c["f1"],
                "edge_iou": c["iou"],
                "edge_ssim":c["ssim"],
                "combined": combined,
                "ssim_verdict":  _change_label(score),
                "edge_verdict":  c["verdict"],
            })
        except Exception as exc:
            results.append({"name": ip.name, "sim": None, "verdict": f"ERROR: {exc}"})

    # ── Print ──────────────────────────────────────────────────────────────
    W = 72
    print()
    print("=" * W)
    print("  FINAL IMAGE SIMILARITY RESULT")
    print("=" * W)

    for r in results:
        if r.get("sim") is None and r.get("combined") is None:
            print(f"  {r['name']}")
            print(f"  Result  : {r['verdict']}")
            print("-" * W)
            continue

        ssim    = r["ssim"]
        f1      = r["edge_f1"]
        combined= r["combined"]

        bar_ssim = "#" * int(ssim     / 2) + "." * (50 - int(ssim     / 2))
        bar_edge = "#" * int(f1       / 2) + "." * (50 - int(f1       / 2))
        bar_comb = "#" * int(combined / 2) + "." * (50 - int(combined / 2))

        print(f"  Input    : {r['name']}")
        print(f"  Output   : {r['output']}")
        print(f"  [{bar_ssim}]")
        print(f"  SSIM Similarity   : {ssim:>6.2f}%   ({r['ssim_verdict']})")
        print(f"  [{bar_edge}]")
        print(f"  Edge (Canny) F1   : {f1:>6.2f}%   ({r['edge_verdict']})")
        print(f"  Edge IoU          : {r['edge_iou']:>6.2f}%")
        print(f"  Edge SSIM         : {r['edge_ssim']:>6.2f}%")
        print(f"  [{bar_comb}]")
        print(f"  COMBINED SCORE    : {combined:>6.2f}%   (SSIM 40% + Edge F1 40% + IoU 20%)")
        print("-" * W)

    print("=" * W)
    print("  100% = identical   |   0% = completely different")
    print("  SSIM     - overall pixel/structure similarity")
    print("  Edge F1  - structural edge match (ignores color & brightness)")
    print("  Combined - weighted final score")
    print("=" * W)
    print()


if __name__ == "__main__":
    compare_folders(
        input_dir="input",
        output_dir="output",
        diff_dir="diff",
        threshold=30,
    )

    segment_diff_folders(
        input_dir="input",
        output_dir="output",
        visual_dir="diff_segments",
        min_area=400,
        tolerance=25,
    )

    final_similarity_report(
        input_dir="input",
        output_dir="output",
    )
