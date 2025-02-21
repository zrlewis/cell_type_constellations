def infer_tree(
        cell_metadata,
        discrete_fields):
    """
    Infer any tree-like relationships (where children in one
    type-field level have one and only one parent in another
    type-field)

    Parameters
    ----------
    cell_metadata:
        a pandas DataFrame of the metadata and annotations
        associated with each cell (obs of an h5ad file)
    discrete_fields:
        a list of columns in cell_metadata by which the cells
        are to be discretely clustered (i.e. the "taxonomic types")

    Returns
    -------
    A dict encoding the child-to-parent relationships between
    fields, i.e.

    {
     child_field0: {
         parent_field0: {
           child0: parent0,
           child1: parent1,
           ....
         },
         parent_field1: {
           child2: parent2,
           child3: parent3,
           ...
         }
         ...
     },
     child_field1: {
       parent_field2: {
         ...
       },
       ...
     },
     ...
    }
    """

    child_to_parent = dict()
    for i0 in range(len(discrete_fields)):
        field0 = discrete_fields[i0]
        for i1 in range(i0+1, len(discrete_fields)):
            field1 = discrete_fields[i1]
            unq_pairs = set(
                [(v0, v1) for v0, v1 in zip(cell_metadata[field0].values,
                                            cell_metadata[field1].values)]
            )

            f0_to_f1 = dict()
            f0_to_f1_valid = True
            f1_to_f0 = dict()
            f1_to_f0_valid = True
            for pair in unq_pairs:
                if pair[0] in f0_to_f1:
                    f0_to_f1_valid = False
                else:
                    f0_to_f1[pair[0]] = pair[1]

                if pair[1] in f1_to_f0:
                    f1_to_f0_valid = False
                else:
                    f1_to_f0[pair[1]] = pair[0]

                f0_to_f1[pair[0]] = pair[1]

            if f0_to_f1_valid:
                if field0 not in child_to_parent:
                    child_to_parent[field0] = dict()
                child_to_parent[field0][field1] = f0_to_f1
            if f1_to_f0_valid:
                if field1 not in child_to_parent:
                    child_to_parent[field1] = dict()
                child_to_parent[field1][field0] = f1_to_f0

    return child_to_parent
