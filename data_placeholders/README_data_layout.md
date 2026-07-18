# Data Layout Placeholder

The transfer package now includes a `data/` folder. This file documents the expected layout in case the data are moved or replaced.

To reproduce the full workflow, create this structure at the project root:

```text
data/
|-- raw/
|   `-- samples/
|       |-- landslide samples.csv
|       `-- landslide_points.csv
`-- processed/
    |-- rasters_cleaned/
    |   `-- 14 cleaned aligned GeoTIFF rasters
    |-- pu_bagging/
    |-- samples/
    |-- patches/
    |-- ssl_unlabeled_indices/
    `-- ssl_pretext_configs/
```

The cleaned raster folder should contain exactly 14 `.tif` files sorted alphabetically and mapped to `factor_01` through `factor_14`.
