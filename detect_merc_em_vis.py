# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.segmentation import find_boundaries


pixel_size_nm = 8
contact_distances_nm = [24, 56]

contact_distances_px = [
    round(d / pixel_size_nm)
    for d in contact_distances_nm
]

data_dir = "."
output_csv = "MERC_per_mitochondrion.csv"

footprint = generate_binary_structure(2, 1)


files = os.listdir(data_dir)

mito_files = [
    f for f in files
    if "mito-prediction" in f.lower()
    and f.lower().endswith((".tif", ".tiff"))
]

er_files = [
    f for f in files
    if "er-prediction" in f.lower()
    and f.lower().endswith((".tif", ".tiff"))
]


if len(mito_files) != 1:
    raise RuntimeError(
        f"Expected one mito-prediction TIFF, found {len(mito_files)}"
    )

if len(er_files) != 1:
    raise RuntimeError(
        f"Expected one er-prediction TIFF, found {len(er_files)}"
    )


mito_path = os.path.join(data_dir, mito_files[0])
er_path = os.path.join(data_dir, er_files[0])


with tifffile.TiffFile(mito_path) as tif:
    n_mito_slices = len(tif.pages)

with tifffile.TiffFile(er_path) as tif:
    n_er_slices = len(tif.pages)


if n_mito_slices != n_er_slices:
    raise ValueError(
        f"Stack depth mismatch: "
        f"Mito={n_mito_slices}, "
        f"ER={n_er_slices}"
    )


n_slices = n_mito_slices


print()
print("Mito:", mito_files[0])
print("ER:", er_files[0])
print("Number of planes:", n_slices)
print("Pixel size:", pixel_size_nm, "nm")
print("Contact distances:", contact_distances_nm, "nm")
print("Expansion radii:", contact_distances_px, "pixels")
print()


surface_total = {}

contact_totals = {
    distance: {}
    for distance in contact_distances_nm
}

first_z = {}
last_z = {}
slice_count = {}


contact_writers = {}

for distance_nm in contact_distances_nm:

    output_path = os.path.join(
        data_dir,
        f"MERC_surface_{distance_nm}nm.tif"
    )

    contact_writers[distance_nm] = tifffile.TiffWriter(
        output_path,
        bigtiff=True
    )


try:

    with tifffile.TiffFile(mito_path) as mito_tif, \
         tifffile.TiffFile(er_path) as er_tif:

        for z in range(n_slices):

            print(
                f"Processing z slice "
                f"{z + 1}/{n_slices}"
            )

            mito = mito_tif.pages[z].asarray()

            er = (
                er_tif.pages[z].asarray()
                > 0
            )


            if mito.shape != er.shape:
                raise ValueError(
                    f"Shape mismatch at z={z}: "
                    f"Mito={mito.shape}, "
                    f"ER={er.shape}"
                )


            surface = find_boundaries(
                mito,
                connectivity=1,
                mode="inner",
                background=0
            )


            surface_labels = mito[
                surface
            ].astype(
                np.int64,
                copy=False
            )


            if surface_labels.size > 0:

                labels, counts = np.unique(
                    surface_labels,
                    return_counts=True
                )

                for label, count in zip(
                    labels,
                    counts
                ):

                    label = int(label)

                    if label == 0:
                        continue

                    surface_total[label] = (
                        surface_total.get(label, 0)
                        + int(count)
                    )


            present_labels = np.unique(mito)

            present_labels = present_labels[
                present_labels != 0
            ]


            for label in present_labels:

                label = int(label)

                if label not in first_z:
                    first_z[label] = z
                    slice_count[label] = 0

                last_z[label] = z
                slice_count[label] += 1


            for distance_nm, distance_px in zip(
                contact_distances_nm,
                contact_distances_px
            ):

                er_expanded = binary_dilation(
                    er,
                    structure=footprint,
                    iterations=distance_px
                )


                contact_mask = (
                    surface
                    & er_expanded
                )


                contact_image = (
                    contact_mask.astype(np.uint8)
                    * 255
                )


                contact_writers[
                    distance_nm
                ].write(
                    contact_image,
                    photometric="minisblack",
                    contiguous=True
                )


                contact_labels = mito[
                    contact_mask
                ].astype(
                    np.int64,
                    copy=False
                )


                if contact_labels.size == 0:
                    continue


                labels, counts = np.unique(
                    contact_labels,
                    return_counts=True
                )


                for label, count in zip(
                    labels,
                    counts
                ):

                    label = int(label)

                    if label == 0:
                        continue

                    contact_totals[
                        distance_nm
                    ][label] = (
                        contact_totals[
                            distance_nm
                        ].get(label, 0)
                        + int(count)
                    )


finally:

    for writer in contact_writers.values():
        writer.close()


results = []


for label in sorted(first_z):

    surface_area = surface_total.get(
        label,
        0
    )


    result = {
        "mito_id": label,
        "first_z": first_z[label],
        "last_z": last_z[label],
        "n_slices": slice_count[label],
        "surface_area_px": surface_area
    }


    for distance_nm in contact_distances_nm:

        contact_area = contact_totals[
            distance_nm
        ].get(
            label,
            0
        )


        if surface_area > 0:

            percent_contact = (
                100.0
                * contact_area
                / surface_area
            )

        else:

            percent_contact = np.nan


        result[
            f"MERC_area_{distance_nm}nm_px"
        ] = contact_area

        result[
            f"MERC_percent_{distance_nm}nm"
        ] = percent_contact


    results.append(result)


df = pd.DataFrame(results)

df = df.sort_values(
    "mito_id"
).reset_index(
    drop=True
)

df.to_csv(
    output_csv,
    index=False
)


print()
print("Analysis complete.")
print(f"Number of mitochondria: {len(df)}")
print(f"Results saved to: {output_csv}")

for distance_nm in contact_distances_nm:
    print(
        f"{distance_nm} nm MERC surface saved to: "
        f"MERC_surface_{distance_nm}nm.tif"
    )