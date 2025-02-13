"""
This module defines a class, TaxonomyFilter, which the rest of the code
uses to manipulate a cell type taxonomy.

Most of its work is actually done by the TaxonomyTree class defined here

https://github.com/AllenInstitute/cell_type_mapper/blob/main/src/cell_type_mapper/taxonomy/taxonomy_tree.py
"""

import anndata
import numpy as np

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree
)


class TaxonomyFilter(object):
    """
    A class to encapsulate and manipulate a cell_type_taxonomy.

    Do not instantiate it by calling __init__. Use one of the
    factory class methods:

    TaxonomyFilter.from_data_release()
    TaxonomyFilter.from_h5ad()
    """

    def __init__(
            self,
            taxonomy_tree):

        self.taxonomy_tree = taxonomy_tree
        self._leaf_tree = self.taxonomy_tree.as_leaves

        self._create_name_to_idx()
        self._create_parentage_to_alias()

    def _create_name_to_idx(self):
        self._name_to_idx = dict()
        self._idx_to_name = dict()
        for level in self.taxonomy_tree.hierarchy:
            self._name_to_idx[level] = {
                'name': dict(),
                'label': dict()
            }
            self._idx_to_name[level] = {
                'name': [],
                'label': []
            }

            for idx, label in enumerate(
                                self.taxonomy_tree.nodes_at_level(level)):

                name = self.taxonomy_tree.label_to_name(
                    level=level,
                    label=label)

                self._name_to_idx[level]['name'][name] = idx
                self._name_to_idx[level]['label'][label] = idx
                self._idx_to_name[level]['name'].append(name)
                self._idx_to_name[level]['label'].append(label)

            self._idx_to_name[level]['name'] = np.array(
                self._idx_to_name[level]['name'])

            self._idx_to_name[level]['label'] = np.array(
                self._idx_to_name[level]['label']
            )

    def _create_parentage_to_alias(self):
        """
        create dict from [level][node] -> np.array of ints indicating
        which aliases belong to that taxon
        """

        alias_to_parentage = _get_alias_to_parentage(self.taxonomy_tree)

        self._parentage_to_alias = dict()
        self._alias_to_idx = dict()

        n_alias = max(list(alias_to_parentage.keys()))+1

        for level in self.taxonomy_tree.hierarchy:
            self._parentage_to_alias[level] = dict()
            self._alias_to_idx[level] = -999*np.ones(n_alias, dtype=int)

        for alias in alias_to_parentage:
            for level in self.taxonomy_tree.hierarchy:
                label = alias_to_parentage[alias][level]['label']
                name = alias_to_parentage[alias][level]['name']
                if label not in self._parentage_to_alias[level]:
                    self._parentage_to_alias[level][label] = []
                    self._parentage_to_alias[level][name] = []
                self._parentage_to_alias[level][label].append(alias)
                self._parentage_to_alias[level][name].append(alias)
                self._alias_to_idx[level][alias] = self._name_to_idx[
                                                         level]['label'][label]

        for level in self.taxonomy_tree.hierarchy:
            node_list = list(self._parentage_to_alias[level])
            for node in node_list:
                self._parentage_to_alias[level][node] = np.array(
                    self._parentage_to_alias[level][node]
                ).astype(int)

    @classmethod
    def from_data_release(
            cls,
            cluster_annotation_path,
            cluster_membership_path,
            hierarchy,
            cell_metadata_path=None):
        """
        Instantiate a TaxonomyFilter from the set of CSV files that
        come with an ABC Atlas data release.

        Parameters
        ----------
        cluster_annotation_path:
            path to the cluster_annotation_term.csv file
        cluster_membership_path:
            path to the cluster_to_cluster_annotation_membership.csv
            file
        hierarchy:
            list of the levels in the cell type taxonomy from
            most gross to most fine
        cell_metadata_path:
            optional path to the cell_metadata.csv file
            (only specified if we are considering a subset
            of the taxonomy defined by cluster_annotation_term.sv
            and cluster_to_cluster_annotation_membership.csv;
            in that case, only cell type taxons annotated to
            contain cells according to cell_metadata.csv will
            be retained)

        Returns
        -------
        a TaxonomyFilter
        """

        taxonomy_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=cell_metadata_path,
            cluster_annotation_path=cluster_annotation_path,
            cluster_membership_path=cluster_membership_path,
            hierarchy=hierarchy,
            do_pruning=True)

        return cls(taxonomy_tree=taxonomy_tree)

    @classmethod
    def from_h5ad(
            cls,
            h5ad_path,
            column_hierarchy,
            cluster_alias_column):
        """
        Instantiate a TaxonomyFilter from an H5AD file.

        Parameters
        ----------
        h5ad_path:
            path to the h5ad file from which data will be read
        column_hierarchy:
            the columns of obs that define the levels of the
            cell type taxonomy, ordered from most gross to most fine
        cluster_alias_column:
            the column in obs that contains the cluster_alias
            (assumed to be an integer)

        Returns
        -------
        a TaxonomyFilter
        """

        raw_taxonomy_tree = TaxonomyTree.from_h5ad(
             h5ad_path=h5ad_path,
             column_hierarchy=column_hierarchy
        )

        adata = anndata.read_h5ad(h5ad_path, backed='r')
        obs = adata.obs
        leaf_level = column_hierarchy[-1]
        cluster_to_alias = dict()
        alias_to_cluster = dict()
        for cluster, alias in zip(
                        obs[leaf_level].values,
                        obs[cluster_alias_column].values):
            alias = str(alias)
            if cluster in cluster_to_alias:
                if cluster_to_alias[cluster] != alias:
                    raise RuntimeError(
                        f'cluster {cluster} maps to several aliases: '
                        f'{alias} and {cluster_to_alias[cluster]}'
                    )
            if alias in alias_to_cluster:
                if alias_to_cluster[alias] != cluster:
                    raise RuntimeError(
                        f'alias {alias} maps to several clusters: '
                        f'{cluster} and {alias_to_cluster[alias]}'
                    )
            cluster_to_alias[cluster] = alias

        name_mapper = dict()
        name_mapper[leaf_level] = dict()

        for cluster in cluster_to_alias:

            name_mapper[leaf_level][cluster] = {
                'alias': cluster_to_alias[cluster]
            }

        tree_data = raw_taxonomy_tree._data
        tree_data['name_mapper'] = name_mapper
        taxonomy_tree = TaxonomyTree(data=tree_data)
        return cls(taxonomy_tree=taxonomy_tree)

    def filter_cells(
            self,
            alias_array,
            level,
            node):
        """
        Given an array of cluster aliases and a specified (level, node)
        pair, return a boolean mask indicating which of the cells
        specified by cluster_aliases are in the specified node, level

        Parameters
        ----------
        alias_array:
            an (n_cells,) array of alias values, each representing
            a different cell and the alias of the cluster to which
            it has been assigned
        level:
            a str. The level of the taxonomy we are considering
        node:
            a str. The specific node with level that we are considering.

        Returns
        -------
        an (n_cells,) array of booleans marked True for every cell that,
        based on the link between cluser aliases and the rest of the
        taxonomy, is annotated to belong to the cell type taxon
        (level, node)
        """

        desired_aliases = self._parentage_to_alias[level][node]

        mask = np.zeros(len(alias_array), dtype=bool)

        for alias in desired_aliases:
            mask[alias_array == alias] = True

        return mask

    def idx_from_label(self, level, node):
        """
        Given a (level, node) pair, what is the index of that
        taxon (for interpreting rows, columns of mixture matrices)
        """
        return self._name_to_idx[level]['label'][node]

    def idx_array_from_alias_array(self, alias_array, level):
        """
        Given an array of cluster aliases, each representing a cell,
        and a level in the cell type taxonomy, what index in that
        level does each cell correspond to (for interpreting
        rows, columns of mixture matrices)
        """
        return self._alias_to_idx[level][alias_array]

    def name_from_idx(self, level, idx):
        """
        Given a level in the cell tyep taxonomy and an index
        at that level (a row or column index in a mixture matrx),
        return a dict with 'name' and 'label' fields.

        'name' is the human-readable name of the corresponding
        cell type taxon

        'label'is the machine-readable, unique name of the corresponding
        cell type taxon
        """
        result = {
            'label': self._idx_to_name[level]['label'][idx],
            'name': self._idx_to_name[level]['name'][idx]
        }
        return result

    def alias_array_from_idx(self, level, idx):
        """
        Given a level of the cell type taxonomy and and index at that
        level (the row or column index of a mixture matrix), return
        the list of cluster aliases corresponding to that index.
        """
        naming = self.name_from_idx(level=level, idx=idx)
        alias = self._parentage_to_alias[level][naming['label']]
        return alias


def _get_alias_to_parentage(taxonomy_tree):
    """
    Take a TaxonomyTree. Return a dict mapping
    cluster alias to a dict encoding the cluster's entire
    parentage.
    """
    results = dict()
    leaf_level = taxonomy_tree.hierarchy[-1]
    for node in taxonomy_tree.nodes_at_level(leaf_level):

        alias = int(
            taxonomy_tree.label_to_name(
                level=leaf_level,
                label=node,
                name_key='alias')
        )

        parentage = taxonomy_tree.parents(
            level=leaf_level,
            node=node)

        formatted = {
            parent_level: {
                'label': parentage[parent_level],
                'name': taxonomy_tree.label_to_name(
                            level=parent_level,
                            label=parentage[parent_level]
                        )
            }
            for parent_level in parentage
        }

        formatted[leaf_level] = {
            'label': node,
            'name': taxonomy_tree.label_to_name(
                        level=leaf_level,
                        label=node
                    )
        }

        results[alias] = formatted

    return results
