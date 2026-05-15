"""Pool an FM's spatial token grid over a polygon's bounding box.

The "per-polygon, per-token" feature extraction works as follows:

    1. Run the FM once on the full chip (256 x 256 px). Get back a token
       tensor of shape [1, N, D] which reshapes to a [H_tok, W_tok, D] grid.
    2. For each polygon, compute its bounding box in chip pixel coordinates.
    3. Map the bbox into the FM's input pixel space (scaled by
       fm_input_size_px / chip_size_px) and then into token indices
       (divided by fm_patch_size_px).
    4. Mean-pool tokens inside the bbox.

This avoids the 128-px-patch context-bleed problem: a 1.28 km patch around
a 0.1 ha polygon is dominated by surrounding land, and the FM ends up
classifying "is there cropland in this big tile?" rather than "does this
specific polygon look like cropland?".

By pooling only the tokens that overlap the polygon, sub-token polygons
collapse to a single token and large polygons average many tokens. The
size-effect we want to surface should appear directly in the size of the
pooled token set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_geom
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry


def polygon_bbox_in_chip_px(
    polygon_wgs84: BaseGeometry, chip_path: Path
) -> tuple[int, int, int, int, int, int]:
    """Return (r_min, c_min, r_max, c_max, chip_h, chip_w) for a polygon
    in the chip's pixel coordinates."""
    with rasterio.open(chip_path) as src:
        poly_proj_geo = transform_geom(
            "EPSG:4326", str(src.crs), polygon_wgs84.__geo_interface__
        )
        poly_shape = shapely_shape(poly_proj_geo)
        minx, miny, maxx, maxy = poly_shape.bounds
        # src.index returns (row, col) for a (x, y) world coordinate.
        r1, c1 = src.index(minx, maxy)
        r2, c2 = src.index(maxx, miny)
        r_min = max(0, min(r1, r2))
        r_max = min(src.height - 1, max(r1, r2))
        c_min = max(0, min(c1, c2))
        c_max = min(src.width - 1, max(c1, c2))
        h, w = src.height, src.width
    return int(r_min), int(c_min), int(r_max), int(c_max), int(h), int(w)


def point_bbox_in_chip_px(
    lon: float, lat: float, chip_path: Path, half_px: int = 8
) -> tuple[int, int, int, int, int, int]:
    """Return a small bbox centered on (lon, lat) in chip pixel coordinates.

    half_px=8 -> 16x16 px window (160 m at S2 10 m GSD), comparable in size
    to a small polygon. Keeps positives and negatives at similar granularity.
    """
    with rasterio.open(chip_path) as src:
        xs, ys = warp_transform("EPSG:4326", str(src.crs), [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        r_min = max(0, r - half_px)
        r_max = min(src.height - 1, r + half_px)
        c_min = max(0, c - half_px)
        c_max = min(src.width - 1, c + half_px)
        h, w = src.height, src.width
    return int(r_min), int(c_min), int(r_max), int(c_max), int(h), int(w)


def pool_tokens_for_bbox(
    tokens_2d: np.ndarray,
    bbox_chip_px: tuple[int, int, int, int],
    chip_size_px: int,
    fm_input_size_px: int,
    fm_patch_size_px: int,
) -> tuple[np.ndarray, int]:
    """Mean-pool token grid over the bbox. Returns (pooled, n_tokens_used).

    tokens_2d: [H_tok, W_tok, D] numpy array of patch tokens.
    bbox_chip_px: (r_min, c_min, r_max, c_max) in chip pixel coords.
    chip_size_px: spatial extent of the chip (assumed square; pass max(h, w)).
    fm_input_size_px: size the FM resizes the chip to internally.
    fm_patch_size_px: FM's per-token patch size in input pixels.

    If the bbox spans less than one token, the single containing token is
    returned (n_tokens_used = 1).
    """
    scale = fm_input_size_px / chip_size_px
    r_min, c_min, r_max, c_max = bbox_chip_px
    rin_min = r_min * scale
    cin_min = c_min * scale
    rin_max = r_max * scale
    cin_max = c_max * scale
    tok_r_min = int(rin_min // fm_patch_size_px)
    tok_c_min = int(cin_min // fm_patch_size_px)
    tok_r_max = int(rin_max // fm_patch_size_px)
    tok_c_max = int(cin_max // fm_patch_size_px)
    h_tok, w_tok = tokens_2d.shape[:2]
    tok_r_min = max(0, min(h_tok - 1, tok_r_min))
    tok_r_max = max(0, min(h_tok - 1, tok_r_max))
    tok_c_min = max(0, min(w_tok - 1, tok_c_min))
    tok_c_max = max(0, min(w_tok - 1, tok_c_max))
    region = tokens_2d[tok_r_min:tok_r_max + 1, tok_c_min:tok_c_max + 1]
    pooled = region.reshape(-1, region.shape[-1]).mean(axis=0)
    n_used = (tok_r_max - tok_r_min + 1) * (tok_c_max - tok_c_min + 1)
    return pooled.astype(np.float32), int(n_used)


def reshape_to_grid(tokens_1d: np.ndarray) -> np.ndarray | None:
    """tokens_1d is [N, D]. If N is a perfect square, reshape to [H, W, D]."""
    n, d = tokens_1d.shape
    side = int(np.sqrt(n))
    if side * side != n:
        return None
    return tokens_1d.reshape(side, side, d)
