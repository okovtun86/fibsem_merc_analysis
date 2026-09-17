# fibsem_merc_analysis
Quantifies mitochondria–ER contact sites (MERCs) from FIB-SEM segmentations using plane-wise surface detection and ER expansion. Reports per-mitochondrion contact fractions at defined distances and exports binary contact surfaces for 3D visualization
# Mitochondria–ER Contact Analysis

Python scripts for quantifying mitochondria–ER contact sites (MERCs) from FIB-SEM segmentations.

The analysis uses a mitochondria instance segmentation and binary ER segmentation exported from Napari as TIFF stacks. Processing is performed plane-wise to accommodate large volumes without loading the complete dataset into memory.

## Analysis

For each 2D plane:

1. A single-pixel-thick outer surface is generated for each mitochondrial instance.
2. The binary ER segmentation is expanded plane-wise to specified distances (default: 24 and 56 nm).
3. Overlap between the mitochondrial surface and expanded ER mask defines the MERC surface.
4. Surface and MERC pixels are assigned to their corresponding mitochondrial instance IDs.
5. Measurements are accumulated across z to calculate the fraction of the total surface of each mitochondrion associated with ER:

`MERC (%) = MERC surface pixels / total mitochondrial surface pixels × 100`

The scripts output one row per mitochondrial instance, including total surface, MERC area, and percent MERC at each distance. Binary TIFF stacks of the MERC surfaces are also generated for visualization in Fiji, Napari, or Imaris.

## Implementations

Two approaches are provided for defining the plane-wise mitochondrial surface.

### `merc_find_boundaries.py`

Uses `skimage.segmentation.find_boundaries(..., mode="inner")` to efficiently identify the inner boundary of each mitochondrial instance.

This provides a fast, label-aware definition of the single-pixel mitochondrial surface.

### `merc_label_erosion.py`

Uses label-aware morphological erosion followed by subtraction from the original mitochondrial segmentation.

A pixel remains after erosion only when the pixels within the structuring element have the same mitochondrial instance ID. This reproduces the plane-wise erosion/subtraction approach while processing all mitochondrial instances simultaneously.

## Input

Place one mitochondria instance-label TIFF stack and one ER-label TIFF stack in the analysis directory.

Filenames should contain:

* `mito-prediction`
* `er-prediction`

The stacks must have matching XY dimensions and numbers of z planes.

The default pixel size is 8 nm and the default ER expansion distances are 24 and 56 nm. These parameters can be changed at the beginning of each script.

## Output

The analysis generates:

* `MERC_per_mitochondrion.csv` — per-mitochondrion MERC measurements across the complete volume
* `MERC_surface_24nm.tif` — binary 24-nm interaction surface
* `MERC_surface_56nm.tif` — binary 56-nm interaction surface

The MERC TIFFs contain the mitochondrial surface pixels satisfying each distance criterion (0 = background, 255 = MERC surface) and can be imported directly into Imaris for 3D visualization.

## Requirements

Python 3 with:

`numpy`, `pandas`, `tifffile`, `scipy`, `scikit-image`

The label-erosion implementation does not require `scikit-image`.
