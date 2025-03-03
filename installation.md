# tips for installation

```
conda create -n constellations python=3.11 anndata cherrypy h5py matplotlib numpy scipy pandas pytest matplotlib
conda activate constellations

pip install -e .

## then check that you can import cell_type_constellations
python
import cell_type_constellations
 ```