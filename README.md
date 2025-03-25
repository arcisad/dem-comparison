# dem-comparison
Utility package for comparing Digital Elevation Models (Currently only REMA vs COP). 
The package enables access to data stored in cloud as well as local copies of dem datasets. 

## Functionality

## Supported DEMS
- Copernicus Global 30m (cop_glo30)
- REMA (2m,10m, ...)

## Usage
### Create mosaicked DEM for bounds from cloud files

```python
from dem_comparison.analysis import analyse_difference_for_interval

import logging
logging.basicConfig(level=logging.INFO)

analyse_difference_for_interval(
    lat_range: range | list[float],
    lon_range: range | list[float],
)
```

## Developer Setup

```bash
git clone ...
conda env create --file environment.yaml
conda activate dem-comparison
pip install -e .
```

Test install

```bash
pytest
```

## Contact
...