import cell_type_constellations.serialization.serialization as serialization


def main():

    h5ad_path = (
    #/Users/zack.lewis/Documents/repositories/cell_type_constellations/data/gpe_cl.h5ad"
    "/Users/zack.lewis/Documents/repositories/cell_type_constellations/data/gpe_ssv4_20250228b_mod.h5ad"
    )

    discrete_fields = [
       "orig.ident",
       "batch",
        "AIT21_type",
        "AIT21_type_consolidated",
        "population"

    ]

    continuous_fields = [
        "aibs.mapping.correlation",
        'xg_X', 'xg_Y', 'xg_Z'
    ]


    dst_path = 'app_data/visualization_hull_test_gpe_ssv4.h5'

    serialization.serialize_from_h5ad(
        h5ad_path=h5ad_path,
        visualization_coords='coords',
        connection_coords_list=['coords','predicted_XY','X_x_scanvi','X_umap_scanvi','X_draw_gr','X_pca'],
        discrete_fields=discrete_fields,
        continuous_fields=continuous_fields,
        leaf_field='AIT21_type',
        discrete_color_map=None,
        dst_path=dst_path,
        tmp_dir='scratch',
        clobber=True,   
        k_nn=25 # default is 15
    )

if __name__ == "__main__":
    main()

"""

        "aibs.class",
        "aibs.subclass",
        "aibs.type",
        "AIT21_type",
        "AIT21_type_consolidated",
        "population"

adata
AnnData object with n_obs × n_vars = 2582 × 32285
    obs: 'orig.idenpertype', 'aibs.type', 'aibs.mapping.correlation', 'tech', 'tech2', 'population', 'sample', 'age', 'facs.pop', 'doublet_score', 'neuronal', 'sex', 'mito.ratio', 'malat1', 'gene_counts', 'umi_counts', 'batch', 'cca_clusters', 'rpca_clusters', 'scANVI_clusters', 'leiden', 'label_pop', 'draw_gr.0', 'draw_gr.1', 'original_leiden_clusters', 'harmony.res.0.8_clusters', 'harmony.res.1_clusters', 'scANVI_clusters_1.6', 'harmony_clusters', 'seurat_clusters', 'harmony_clusters_0.8', 'harmony_clusters_1', 'harmony_clusters_1.6', 'gpe_subset_good_clust_reclustered', 'subset_harmony_clusters', 'subset_harmony_clusters0.8', 'subset_harmony_clusters_0.8', 'subset_harmony_clusters_0.5', 'subset_harmony_clusters_0.3', 'scANVI_clusters_sub', 'phenotype', 'subcluster_id', 'subcluster', 'nCluster_0.5', 'nCluster_0.3', 'cluster_name', 'cluster_number', 'X', 'Y', 'Z', 'rf_X', 'rf_Y', 'rf_Z', 'xg_X', 'xg_Y', 'xg_Z', 'AIT21_type', 'AIT21_type_consolidated'
    unt', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'aibs.class_broad', 'aibs.class', 'aibs.subclass', 'aibs.sus: 'altExperiments', 'commands', 'version'
    obsm: 'X_draw_gr', 'X_harmony', 'X_pca', 'X_subset_harmony', 'X_umap.harmony', 'X_umap.harmony.sub', 'X_umap.scanvi', 'X_umap.scanvi.sub', 'X_umap.sub_good.pca', 'X_umap_gpe_nn_sub_scanpy', 'X_umap_gpe_nn_sub_scanpy_reint', 'X_umap_scanvi', 'X_x_scanvi', 'coords', 'predicted_XY'
    layers: 'RNA_counts', 'RNA_data'
    obsp: 'RNA_nn', 'RNA_snn'


"""