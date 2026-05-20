"""
assemble_hyperstacks.py

Assembles folders of individual microscopy tiff files into per-region ImageJ
hyperstacks that Fiji can open with Z, T, and C sliders.

===============================================================================
REQUIRED FILE STRUCTURE (default output from squid)
===============================================================================

    datadir/
        acquisition parameters.json       (optional but recommended)
        0/                                 time point folders, named by index
            R0_0_0_BF_LED_matrix_full.tiff
            R0_0_1_BF_LED_matrix_full.tiff
            R0_0_0_Fluorescence_488_nm_Ex.tiff
            ...
        1/
            ...
        26/
            ...

TIME POINT FOLDERS
    Each sub-folder must be named with an integer (0, 1, 2, ...).
    Non-integer folders (e.g. the output files themselves) are ignored.

TIFF FILENAME FORMAT
    Every tiff must match:

        <region>_<fov>_<zpos>_<channel>.tiff

    region  : R followed by an integer,  e.g. R0, R1, R4
    fov     : field-of-view index (integer); currently only 0 is supported
    zpos    : z-slice index (integer),       e.g. 0, 1, ..., 19
    channel : arbitrary string,              e.g. BF_LED_matrix_full
                                                  Fluorescence_488_nm_Ex

    Example: R1_0_4_Fluorescence_488_nm_Ex.tiff
             ^  ^ ^  ^
             |  | |  channel
             |  | z-slice 4
             |  fov 0
             region R1

ACQUISITION PARAMETERS JSON  (optional)
    If present at datadir/acquisition parameters.json, the following keys
    are read for physical calibration:

        dz(um)                    z step size in micrometers
        dt(s)                     time interval in seconds (0 is treated as 1)
        sensor_pixel_size_um      camera sensor pixel size in micrometers
        objective.magnification   objective magnification (e.g. 20.0)

    XY pixel size is computed as sensor_pixel_size_um / magnification.
    Override with --pixel-size-um if needed.
    If the file is absent, all axes default to 1 unit per pixel.

===============================================================================
USAGE EXAMPLES
===============================================================================

    # Basic -- all regions, full resolution
    python assemble_hyperstacks.py --datadir /path/to/timelapse

    # Process only R0 and R2
    python assemble_hyperstacks.py --datadir /path/to/timelapse --regions R0 R2

    # Downsample XY by 2x (~4x smaller file, pixel binning)
    python assemble_hyperstacks.py --datadir /path/to/timelapse --downsample 2

    # Override the calculated XY pixel size
    python assemble_hyperstacks.py --datadir /path/to/timelapse --pixel-size-um 0.376

OUTPUT
    One file per region, written to datadir:
        R0_hyperstack.tiff
        R0_hyperstack_ds2x.tiff   (if --downsample 2 was used)

    Open in Fiji -- it will automatically detect the hyperstack and show
    sliders for Z, T, and C. Physical spacing (um) is embedded so
    Image > Properties will show calibrated axes.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile
from skimage.transform import downscale_local_mean


# ---------------------------------------------------------------------------
# JSON acquisition parameters
# ---------------------------------------------------------------------------

def load_acquisition_params(datadir: Path) -> dict:
    """
    Load 'acquisition parameters.json' from datadir if present.
    Returns a dict with keys: dx_mm, dy_mm, dz_um, dt_s (all floats).
    Missing keys fall back to sensible defaults (1.0 um / 1.0 s).
    """
    json_path = datadir / "acquisition parameters.json"
    params = {}

    if not json_path.exists():
        print("  No 'acquisition parameters.json' found -- using default spacing (1 um/px).")
        return {'dx_mm': None, 'dy_mm': None, 'dz_um': 1.0, 'dt_s': 1.0}

    with open(json_path, 'r') as f:
        raw = json.load(f)

    params['dx_mm'] = raw.get('dx(mm)', None)
    params['dy_mm'] = raw.get('dy(mm)', None)
    params['dz_um'] = raw.get('dz(um)', 1.0)
    params['dt_s']  = raw.get('dt(s)',  1.0)

    # dt=0 is stored when there is only one timepoint; treat as 1.0 to avoid
    # Fiji showing a zero time interval (which it handles poorly)
    if params['dt_s'] == 0:
        params['dt_s'] = 1.0

    # Compute per-pixel XY size on the sample from sensor pixel size and magnification
    sensor_px_um   = raw.get('sensor_pixel_size_um', None)
    magnification  = None
    obj = raw.get('objective', {})
    if isinstance(obj, dict):
        magnification = obj.get('magnification', None)

    if sensor_px_um is not None and magnification is not None and magnification > 0:
        params['pixel_size_um'] = sensor_px_um / magnification
    else:
        params['pixel_size_um'] = None

    print(f"  Acquisition params: dz={params['dz_um']} um, dt={params['dt_s']} s, "
          f"pixel size={params['pixel_size_um']} um/px "
          f"(sensor={sensor_px_um} um, mag={magnification}x)")

    return params


def xy_resolution_tags(dx_mm: float, dy_mm: float, xy_downsample: int):
    """
    Convert dx/dy in mm to pixels-per-mm for the TIFF XResolution/YResolution
    tags (which Fiji reads as physical pixel size).

    Returns (x_res, y_res) as (pixels_per_mm, pixels_per_mm) floats,
    already scaled for any XY downsampling factor.
    """
    if dx_mm is None or dy_mm is None:
        return None, None

    # Stage step in mm is the distance moved per image pixel at the camera.
    # But the actual pixel size on the sample depends on the sensor pixel size
    # and magnification -- dx(mm) here represents the field shift between
    # adjacent positions, NOT the per-pixel spacing.
    #
    # Per-pixel spacing in mm = dx(mm) / Nx ... but Nx=1 for a single FOV.
    # The true per-pixel spacing comes from: sensor_pixel_size / magnification.
    # Since we don't have that here, we store dx as a field-of-view note only.
    #
    # Instead we return None so Fiji uses pixel units, which is honest.
    # If you know your per-pixel spacing in um, pass it via --pixel-size-um.
    return None, None


# ---------------------------------------------------------------------------
# File discovery and axis inference
# ---------------------------------------------------------------------------

def parse_filename(filepath: Path):
    """
    Parse a filename of the form region_0_zpos_channel.tiff
    Returns (region, zpos, channel) or None if it doesn't match.
    """
    name = filepath.stem
    match = re.match(r'^(R\d+)_(\d+)_(\d+)_(.+)$', name)
    if match:
        region  = match.group(1)
        zpos    = int(match.group(3))
        channel = match.group(4)
        return region, zpos, channel
    return None


def discover_files(datadir: Path):
    """
    Walk all time point subdirectories and collect file info.
    Returns a dict: region -> list of (timepoint, zpos, channel, filepath)
    """
    region_files = defaultdict(list)

    for time_folder in sorted(datadir.iterdir()):
        if not time_folder.is_dir():
            continue
        try:
            timepoint = int(time_folder.name)
        except ValueError:
            continue  # skip non-numeric folders

        for filepath in sorted(time_folder.glob('*.tiff')):
            result = parse_filename(filepath)
            if result is None:
                continue
            region, zpos, channel = result
            region_files[region].append((timepoint, zpos, channel, filepath))

    return region_files


def get_axes_info(file_list):
    timepoints = sorted(set(t for t, z, c, _ in file_list))
    zpositions = sorted(set(z for t, z, c, _ in file_list))
    channels   = sorted(set(c for t, z, c, _ in file_list))
    return timepoints, zpositions, channels


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def read_image_2d(filepath: Path) -> np.ndarray:
    """Read a tiff and ensure it comes back as a 2D (Y, X) array."""
    img = tifffile.imread(str(filepath))
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2D image after squeezing, got shape {img.shape} for {filepath}"
        )
    return img


def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 2D image by averaging (factor x factor) pixel blocks.
    Uses local mean averaging (pixel binning) to preserve signal integrity.
    Output is cast back to the original dtype.
    """
    downscaled = downscale_local_mean(img.astype(np.float32), (factor, factor))
    return downscaled.astype(img.dtype)


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def assemble_region(region: str, file_list: list, outdir: Path,
                    acq_params: dict, xy_downsample: int = 1,
                    pixel_size_um: float = None):
    """
    Load all tiffs for one region and save as an ImageJ hyperstack (TZCYX order).

    Physical spacing embedded in the output tiff:
      - XY pixel size: from --pixel-size-um if provided, else left as pixel units
      - Z step:        dz(um) from acquisition parameters.json
      - Time interval: dt(s)  from acquisition parameters.json

    Args:
        region:        Region label, e.g. 'R1'
        file_list:     List of (timepoint, zpos, channel, filepath) tuples
        outdir:        Directory to write the output tiff
        acq_params:    Dict from load_acquisition_params()
        xy_downsample: Integer XY binning factor (1 = no downsampling)
        pixel_size_um: Override per-pixel XY size in um (e.g. from sensor size /
                       magnification). If None, XY axes are stored in pixel units.
    """
    timepoints, zpositions, channels = get_axes_info(file_list)

    T = len(timepoints)
    C = len(channels)
    Z = len(zpositions)

    t_idx = {t: i for i, t in enumerate(timepoints)}
    z_idx = {z: i for i, z in enumerate(zpositions)}
    c_idx = {c: i for i, c in enumerate(channels)}

    # Probe first file for spatial dims and dtype
    sample = read_image_2d(file_list[0][3])
    Y, X   = sample.shape
    dtype  = sample.dtype

    Y_out = Y // xy_downsample
    X_out = X // xy_downsample
    ds_note = (f" -> downsampled {xy_downsample}x to {Y_out}x{X_out}"
               if xy_downsample > 1 else "")
    print(f"  Array dims -- T:{T}  C:{C}  Z:{Z}  Y:{Y}  X:{X}  dtype:{dtype}{ds_note}")

    # Build TCZYX array, then swap to TZCYX for tifffile
    stack = np.zeros((T, C, Z, Y_out, X_out), dtype=dtype)

    for timepoint, zpos, channel, filepath in file_list:
        ti = t_idx[timepoint]
        zi = z_idx[zpos]
        ci = c_idx[channel]
        img = read_image_2d(filepath)
        if xy_downsample > 1:
            img = downsample_image(img, xy_downsample)
        stack[ti, ci, zi] = img

    filled   = len(file_list)
    expected = T * C * Z
    if filled < expected:
        print(f"  Warning: {expected - filled} missing file(s) out of {expected} -- "
              f"missing slots will be zero.")

    # tifffile ImageJ mode expects TZCYX axis order
    stack_tzcyx = np.swapaxes(stack, 1, 2)

    # ------------------------------------------------------------------
    # Physical spacing
    # ------------------------------------------------------------------
    dz_um = acq_params.get('dz_um', 1.0)
    dt_s  = acq_params.get('dt_s',  1.0)

    # Effective XY pixel size in um after downsampling.
    # Priority: CLI --pixel-size-um > calculated from JSON > None (pixel units)
    if pixel_size_um is not None:
        eff_px_um = pixel_size_um * xy_downsample
    elif acq_params.get('pixel_size_um') is not None:
        eff_px_um = acq_params['pixel_size_um'] * xy_downsample
    else:
        eff_px_um = None

    # TIFF resolution tags store pixels-per-unit as an integer rational (num, den).
    # Fiji reads these for XY calibration. resolutionunit=MICROMETER tells Fiji the unit.
    # We encode 1/eff_px_um (px/um) as a reduced integer fraction to avoid overflow.
    if eff_px_um is not None and eff_px_um > 0:
        from math import gcd
        scale = 100_000
        num = scale
        den = round(eff_px_um * scale)
        g   = gcd(num, den)
        num //= g
        den //= g
        xy_resolution   = ((num, den), (num, den))   # (px/um for X, px/um for Y)
        resolution_unit = tifffile.RESUNIT.MICROMETER
        print(f"  XY pixel size: {eff_px_um:.6f} um/px  "
              f"(resolution tag: {num}/{den} px/um)")
    else:
        xy_resolution   = None
        resolution_unit = None
        if pixel_size_um is None:
            print("  XY pixel size: not set (use --pixel-size-um to calibrate, "
                  "or set manually in Fiji via Image > Properties)")

    # ImageJ metadata dict -- do NOT include 'axes' key here
    imagej_metadata = {
        'spacing':   dz_um,   # Z step in um
        'finterval': dt_s,    # time interval in seconds
        'unit':      'um',
    }

    ds_suffix = f"_ds{xy_downsample}x" if xy_downsample > 1 else ""
    out_path  = outdir / f"{region}_hyperstack{ds_suffix}.tiff"
    print(f"  Saving -> {out_path}")

    write_kwargs = dict(
        imagej=True,
        metadata=imagej_metadata,
    )
    if xy_resolution is not None:
        write_kwargs['resolution']     = xy_resolution
        write_kwargs['resolutionunit'] = resolution_unit

    tifffile.imwrite(str(out_path), stack_tzcyx, **write_kwargs)
    print(f"  Done. Output shape (TZCYX): {stack_tzcyx.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble microscopy tiffs into Fiji hyperstacks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
XY pixel size note:
  dx(mm) in acquisition parameters.json is the stage step between FOV positions,
  not the per-pixel size on the sample. To get per-pixel size use:
      pixel_size_um = sensor_pixel_size_um / magnification
  e.g. for the 20x objective with 7.52 um sensor pixels:
      pixel_size_um = 7.52 / 20 = 0.376 um/px
  Pass this via --pixel-size-um 0.376
        """
    )
    parser.add_argument('--datadir', required=True,
                        help='Path to the timelapse directory.')
    parser.add_argument('--regions', nargs='*', default=None,
                        help='Regions to process (e.g. R0 R1). Default: all.')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Integer XY downsampling factor (default: 1). '
                             'E.g. --downsample 2 halves width and height (~4x smaller file).')
    parser.add_argument('--pixel-size-um', type=float, default=None,
                        dest='pixel_size_um',
                        help='Per-pixel XY size in um on the sample (before any downsampling). '
                             'For the 20x / 7.52 um sensor: 7.52 / 20 = 0.376. '
                             'If omitted, XY axes are stored in pixel units.')
    args = parser.parse_args()

    if args.downsample < 1:
        raise ValueError("--downsample must be a positive integer.")

    datadir = Path(args.datadir)

    # Load acquisition parameters once for all regions
    print(f"Loading acquisition parameters from {datadir} ...")
    acq_params = load_acquisition_params(datadir)
    print()

    print(f"Scanning {datadir} for tiff files ...")
    region_files = discover_files(datadir)

    if not region_files:
        print("No matching .tiff files found. Check directory structure and filename format.")
        return

    regions_to_process = args.regions if args.regions else sorted(region_files.keys())
    print(f"Regions found:      {sorted(region_files.keys())}")
    print(f"Regions to process: {regions_to_process}")
    if args.downsample > 1:
        print(f"XY downsampling:    {args.downsample}x  "
              f"(~{args.downsample**2}x smaller file)")
    print()

    for region in regions_to_process:
        if region not in region_files:
            print(f"Region {region} not found, skipping.")
            continue

        file_list = region_files[region]
        timepoints, zpositions, channels = get_axes_info(file_list)
        print(f"Region {region}: {len(timepoints)} timepoint(s), "
              f"{len(zpositions)} z slice(s), {len(channels)} channel(s)")
        print(f"  Channels: {channels}")

        assemble_region(
            region, file_list, datadir, acq_params,
            xy_downsample=args.downsample,
            pixel_size_um=args.pixel_size_um,
        )
        print()

    print("All done! Open the output .tiff files in Fiji -- "
          "they should automatically open as hyperstacks with Z, T, and C sliders.")


if __name__ == '__main__':
    main()
