# Troubleshooting issues

_Zack Lewis_

- Previous versions of this code expected certain `obs` are present. In particular, there must be a column in `obs` that provides unique integers for each taxonomic cluster. 
- To generate that column, you can use some of the scripts below.
- This should no longer be needed. 

```
import sys
import anndata
import re

def check_h5ad(h5ad_path):
    ad = anndata.read_h5ad(h5ad_path)
    obs = ad.obs
    return obs

def list_obs_keys(h5ad_path):
    ad = anndata.read_h5ad(h5ad_path)
    keys = ad.obs_keys
    return keys

# check if there is a column called `cl` in obs
def check_cl(h5ad_path):
    obs = check_h5ad(h5ad_path)
    if 'cl' in obs.columns:
        return '`cl` column found'
    else:
        return '`cl` column not found. `cl` is an integer, and is a unique cluster identifier'
    
# create a cl column in the anndata.obs and save a new h5ad file
# use the integer from the `aibs.type` column
def add_cl(h5ad_path, new_h5ad_path):
    ad = anndata.read_h5ad(h5ad_path)
    # Apply slicing on each value in the 'aibs.type' column
    ad.obs['cl'] = ad.obs['aibs.type'].str[:4].astype(int)
    ad.write(new_h5ad_path)
```

- I also encountered an error where the `visualization_coords` in the `cell_type_constellations.cli.serialize_svg_data_from_h5ad` command expects a 2D set of coordinates. 
- This also should no longer be an issue, but the same approach can be used to extract different axes from the dimensional reductions. 

```
import anndata

# make a new coordinate column in the anndata.obsm and save a new h5ad file
# use the first two UMAP dimensions
# adapt the name of the original obsm slot and new coordinates as needed
def add_coords(h5ad_path, new_h5ad_path):
    adata = anndata.read_h5ad(h5ad_path)
    adata.obsm['coords'] = adata.obsm['X_scANVI'][:, :2]
    adata.write(new_h5ad_path)

```

- Create a new `obsm` layer from a combination of two `obs` columns.
- These can end up being new latent variables for edge calculation.

```
import anndata

# make a new latent variable column in the anndata.obsm and save a new h5ad file
# use the first two UMAP dimensions
# adapt the name of the original obsm slot and new coordinates as needed
def add_coords(h5ad_path, new_h5ad_path):
    adata = anndata.read_h5ad(h5ad_path)
     # Extract 'xg_X' and 'xg_Y' from obs and store in obsm as a NumPy array
    adata.obsm['predicted_XY'] = adata.obs[['xg_X', 'xg_Y']].to_numpy()
    adata.write(new_h5ad_path)

```