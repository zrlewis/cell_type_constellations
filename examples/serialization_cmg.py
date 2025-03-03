import cell_type_constellations.serialization.serialization as serialization


def main():

    h5ad_path = (
    "/Users/zack.lewis/Downloads/cmg.neural.20250227.h5ad"
    )

    discrete_fields = [
        "cell_description",
        "cell_type",
        "treatment_binned",
        "tech",
        "sex",
        "batch"
    ]

    continuous_fields = [
        "nCount_RNA",
        "nFeature_RNA",
        "mito.ratio",
        "percent_mito_ribo",

    ]


    dst_path = 'app_data/visualization_hull_cmg.h5'

    serialization.serialize_from_h5ad(
        h5ad_path=h5ad_path,
        visualization_coords='X_umap.harmony',
        connection_coords_list=['X_harmony', 'X_pca', 'X_umap.harmony'],
        discrete_fields=discrete_fields,
        continuous_fields=continuous_fields,
        leaf_field='cell_description',
        discrete_color_map=None,
        dst_path=dst_path,
        tmp_dir='scratch',
        clobber=True
    )

if __name__ == "__main__":
    main()
