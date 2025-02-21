import cell_type_constellations.serialization.serialization as serialization


def main():

    h5ad_path = (
    "/Users/scott.daniel/KnowledgeBase/cell_type_constellations/data/HY-EA/HY_v1.h5ad"
    )

    discrete_fields = [
        "class_label",
        "subclass_label",
        "supertype_label",
        "cluster_id_label",
        "nt_type_label"
    ]

    continuous_fields = [
        "Attack-M vs Control-M",
        "Sexual-M vs Control-M",
        "Predator Fear vs Control-Fear",
        "Fasting vs Control-Feeding",
        "Fasting + Re-fed vs Control-Feeding"
    ]


    dst_path = 'app_data/visualization_hull_test.h5'

    serialization.serialize_from_h5ad(
        h5ad_path=h5ad_path,
        visualization_coords='X_umap',
        connection_coords_list=['X_umap', 'X_scVI'],
        discrete_fields=discrete_fields,
        continuous_fields=continuous_fields,
        leaf_field='cluster_id_label',
        discrete_color_map=None,
        dst_path=dst_path,
        tmp_dir='scratch',
        clobber=True
    )

if __name__ == "__main__":
    main()
