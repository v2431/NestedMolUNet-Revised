"""
Advanced splitting strategies for DTI datasets.

This module provides implementations of various splitting strategies for drug-target interaction
datasets, including scaffold split and cold-start splits (cold protein, cold compound, cold pair).
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Generate a Murcko scaffold from a SMILES string.
    
    Args:
        smiles: SMILES string of the molecule
        include_chirality: Whether to include chirality information in the scaffold
        
    Returns:
        The SMILES string of the scaffold
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_split(df, smiles_col='smiles', seed=42, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split a dataset by molecular scaffold.
    
    Args:
        df: DataFrame containing the dataset
        smiles_col: Name of the column containing SMILES strings
        seed: Random seed
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    np.testing.assert_almost_equal(train_size + valid_size + test_size, 1.0)
    
    scaffolds = {}
    unassigned = []
    for idx, smiles in zip(df.index, df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            unassigned.append(idx)
            continue
        scaffolds.setdefault(scaffold, []).append(idx)
            
    scaffold_sets = [scaffold_set for _, scaffold_set in sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)]
    
    train_cutoff = train_size * len(df)
    valid_cutoff = (train_size + valid_size) * len(df)
    
    train_idx = []
    valid_idx = []
    test_idx = []
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= valid_cutoff - len(train_idx):
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    if unassigned:
        print(f"Warning: {len(unassigned)} samples could not be assigned to a scaffold")
        train_unassigned = unassigned[:int(len(unassigned) * train_size)]
        valid_unassigned = unassigned[int(len(unassigned) * train_size):int(len(unassigned) * (train_size + valid_size))]
        test_unassigned = unassigned[int(len(unassigned) * (train_size + valid_size)):]
        
        train_idx.extend(train_unassigned)
        valid_idx.extend(valid_unassigned)
        test_idx.extend(test_unassigned)
    
    return train_idx, valid_idx, test_idx


def cold_protein_split(df, protein_col='target_sequence', seed=42, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split a dataset such that test set contains proteins not seen during training.
    
    Args:
        df: DataFrame containing the dataset
        protein_col: Name of the column containing protein sequences
        seed: Random seed
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    np.testing.assert_almost_equal(train_size + valid_size + test_size, 1.0)
    rng = np.random.RandomState(seed)
    
    # Get unique proteins
    unique_proteins = df[protein_col].unique()
    rng.shuffle(unique_proteins)
    
    # Split proteins
    n_proteins = len(unique_proteins)
    train_proteins = unique_proteins[:int(n_proteins * train_size)]
    valid_proteins = unique_proteins[int(n_proteins * train_size):int(n_proteins * (train_size + valid_size))]
    test_proteins = unique_proteins[int(n_proteins * (train_size + valid_size)):]
    
    # Get indices
    train_idx = df[df[protein_col].isin(train_proteins)].index.tolist()
    valid_idx = df[df[protein_col].isin(valid_proteins)].index.tolist()
    test_idx = df[df[protein_col].isin(test_proteins)].index.tolist()
    
    return train_idx, valid_idx, test_idx


def cold_compound_split(df, smiles_col='smiles', seed=42, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split a dataset such that test set contains compounds not seen during training.
    
    Args:
        df: DataFrame containing the dataset
        smiles_col: Name of the column containing SMILES strings
        seed: Random seed
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    np.testing.assert_almost_equal(train_size + valid_size + test_size, 1.0)
    rng = np.random.RandomState(seed)
    
    # Get unique compounds
    unique_compounds = df[smiles_col].unique()
    rng.shuffle(unique_compounds)
    
    # Split compounds
    n_compounds = len(unique_compounds)
    train_compounds = unique_compounds[:int(n_compounds * train_size)]
    valid_compounds = unique_compounds[int(n_compounds * train_size):int(n_compounds * (train_size + valid_size))]
    test_compounds = unique_compounds[int(n_compounds * (train_size + valid_size)):]
    
    # Get indices
    train_idx = df[df[smiles_col].isin(train_compounds)].index.tolist()
    valid_idx = df[df[smiles_col].isin(valid_compounds)].index.tolist()
    test_idx = df[df[smiles_col].isin(test_compounds)].index.tolist()
    
    return train_idx, valid_idx, test_idx


def cold_pair_split(df, protein_col='target_sequence', smiles_col='smiles', seed=42, 
                    train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split a dataset such that test set contains pairs where either the protein or
    the compound or both were not seen during training.
    
    Args:
        df: DataFrame containing the dataset
        protein_col: Name of the column containing protein sequences
        smiles_col: Name of the column containing SMILES strings
        seed: Random seed
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    np.testing.assert_almost_equal(train_size + valid_size + test_size, 1.0)
    rng = np.random.RandomState(seed)
    
    pair_id = df[protein_col].astype(str) + '___' + df[smiles_col].astype(str)
    
    unique_pairs = pair_id.dropna().unique()
    rng.shuffle(unique_pairs)
    
    n_pairs = len(unique_pairs)
    train_pairs = set(unique_pairs[:int(n_pairs * train_size)])
    valid_pairs = set(unique_pairs[int(n_pairs * train_size):int(n_pairs * (train_size + valid_size))])
    test_pairs = set(unique_pairs[int(n_pairs * (train_size + valid_size)):])
    
    train_idx = df.index[pair_id.isin(train_pairs)].tolist()
    valid_idx = df.index[pair_id.isin(valid_pairs)].tolist()
    test_idx = df.index[pair_id.isin(test_pairs)].tolist()
    
    return train_idx, valid_idx, test_idx


def blind_start_split(df, protein_col='target_sequence', smiles_col='smiles', seed=42,
                      train_entity_frac=0.6, test_entity_frac=0.4, canonicalize_smiles=False):
    """
    Blind Start CV split (as defined in Nat. Commun. s41467-025-61745-7).
    
    This split creates a double-cold test set where neither compounds nor proteins
    have been seen during training.
    
    Strategy:
    - Randomly split unique compounds into C_train (train_entity_frac) and C_test (test_entity_frac)
    - Randomly split unique proteins into P_train (train_entity_frac) and P_test (test_entity_frac)
    - Train: pairs with (c in C_train) AND (p in P_train)
    - Test:  pairs with (c in C_test)  AND (p in P_test)  [double-cold]
    - Val:   remaining pairs (cross blocks): (C_train x P_test) U (C_test x P_train)
    
    Args:
        df: DataFrame containing the dataset
        protein_col: Name of the column containing protein sequences
        smiles_col: Name of the column containing SMILES strings
        seed: Random seed
        train_entity_frac: Fraction of unique entities for training (default 0.6)
        test_entity_frac: Fraction of unique entities for testing (default 0.4)
        canonicalize_smiles: Whether to canonicalize SMILES to reduce leakage
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    np.testing.assert_almost_equal(train_entity_frac + test_entity_frac, 1.0)
    
    rng = np.random.RandomState(seed)
    
    work_df = df.copy()
    
    if canonicalize_smiles:
        def _canon(s):
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                return str(s)
            return Chem.MolToSmiles(m, canonical=True)
        work_df[smiles_col] = work_df[smiles_col].astype(str).map(_canon)
    
    unique_compounds = work_df[smiles_col].dropna().unique()
    unique_proteins = work_df[protein_col].dropna().unique()
    
    rng.shuffle(unique_compounds)
    rng.shuffle(unique_proteins)
    
    n_c = len(unique_compounds)
    n_p = len(unique_proteins)
    
    c_cut = int(n_c * train_entity_frac)
    p_cut = int(n_p * train_entity_frac)
    
    C_train = set(unique_compounds[:c_cut])
    C_test = set(unique_compounds[c_cut:])
    
    P_train = set(unique_proteins[:p_cut])
    P_test = set(unique_proteins[p_cut:])
    
    c_in_train = work_df[smiles_col].isin(C_train)
    p_in_train = work_df[protein_col].isin(P_train)
    c_in_test = work_df[smiles_col].isin(C_test)
    p_in_test = work_df[protein_col].isin(P_test)
    
    train_mask = c_in_train & p_in_train
    test_mask = c_in_test & p_in_test
    valid_mask = ~(train_mask | test_mask)
    
    train_idx = work_df.index[train_mask].tolist()
    valid_idx = work_df.index[valid_mask].tolist()
    test_idx = work_df.index[test_mask].tolist()
    
    return train_idx, valid_idx, test_idx


def _canon_smiles_safe(s):
    """Safely canonicalize a SMILES string."""
    s = str(s)
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else s


def summarize_split(df, train_idx, valid_idx, test_idx,
                    protein_col='target_sequence', smiles_col='smiles',
                    canonicalize_smiles_for_checks=True):
    """
    Return a dict of split stats + leakage checks.
    
    Args:
        df: Original DataFrame
        train_idx, valid_idx, test_idx: Split indices
        protein_col: Name of protein column
        smiles_col: Name of SMILES column
        canonicalize_smiles_for_checks: Whether to canonicalize SMILES for overlap checks
        
    Returns:
        dict: Statistics including sample counts, entity counts, and overlap checks
    """
    train_idx = np.asarray(train_idx).tolist()
    valid_idx = np.asarray(valid_idx).tolist()
    test_idx = np.asarray(test_idx).tolist()
    
    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]
    test_df = df.loc[test_idx]

    def comp_set(xdf):
        if canonicalize_smiles_for_checks:
            return set(xdf[smiles_col].map(_canon_smiles_safe))
        return set(xdf[smiles_col].astype(str))

    train_p = set(train_df[protein_col].astype(str))
    valid_p = set(valid_df[protein_col].astype(str))
    test_p = set(test_df[protein_col].astype(str))

    train_c = comp_set(train_df)
    valid_c = comp_set(valid_df)
    test_c = comp_set(test_df)

    stats = {
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
        "frac_train": len(train_idx) / max(len(df), 1),
        "frac_valid": len(valid_idx) / max(len(df), 1),
        "frac_test": len(test_idx) / max(len(df), 1),
        "train_unique_proteins": len(train_p),
        "valid_unique_proteins": len(valid_p),
        "test_unique_proteins": len(test_p),
        "train_unique_compounds": len(train_c),
        "valid_unique_compounds": len(valid_c),
        "test_unique_compounds": len(test_c),
        "train_test_protein_overlap": len(train_p & test_p),
        "train_test_compound_overlap": len(train_c & test_c),
        "train_valid_protein_overlap": len(train_p & valid_p),
        "train_valid_compound_overlap": len(train_c & valid_c),
    }
    return stats


def assert_blind_start_ok(stats, min_test_samples=1, max_test_frac=None):
    """
    Raise AssertionError if blind-start constraints aren't met.
    
    Args:
        stats: Statistics dict from summarize_split()
        min_test_samples: Minimum required test samples
        max_test_frac: Maximum allowed test fraction (optional)
        
    Raises:
        AssertionError: If constraints are violated
    """
    assert stats["n_test"] >= min_test_samples, \
        f"Test too small: {stats['n_test']} < {min_test_samples}"
    assert stats["train_test_protein_overlap"] == 0, \
        f"Protein leakage train<->test: {stats['train_test_protein_overlap']}"
    assert stats["train_test_compound_overlap"] == 0, \
        f"Compound leakage train<->test: {stats['train_test_compound_overlap']}"
    if max_test_frac is not None:
        assert stats["frac_test"] <= max_test_frac, \
            f"Test too large: {stats['frac_test']:.3f} > {max_test_frac}"


def blind_start_split_with_retry(
    df,
    protein_col='target_sequence',
    smiles_col='smiles',
    seed=42,
    train_entity_frac=0.6,
    test_entity_frac=0.4,
    canonicalize_smiles=True,
    min_test_samples=200,
    max_tries=50,
    verbose=True
):
    """
    Blind start split with automatic retry if constraints not met.
    
    Args:
        df: DataFrame containing the dataset
        protein_col: Name of the column containing protein sequences
        smiles_col: Name of the column containing SMILES strings
        seed: Initial random seed
        train_entity_frac: Fraction of unique entities for training
        test_entity_frac: Fraction of unique entities for testing
        canonicalize_smiles: Whether to canonicalize SMILES
        min_test_samples: Minimum required test samples
        max_tries: Maximum number of retry attempts
        verbose: Whether to print progress
        
    Returns:
        train_idx, valid_idx, test_idx, stats: Split indices and statistics dict
        
    Raises:
        RuntimeError: If unable to generate valid split after max_tries
    """
    last_stats = None
    for k in range(max_tries):
        cur_seed = seed + k

        train_idx, valid_idx, test_idx = blind_start_split(
            df,
            protein_col=protein_col,
            smiles_col=smiles_col,
            seed=cur_seed,
            train_entity_frac=train_entity_frac,
            test_entity_frac=test_entity_frac,
            canonicalize_smiles=canonicalize_smiles
        )

        stats = summarize_split(
            df, train_idx, valid_idx, test_idx,
            protein_col=protein_col, smiles_col=smiles_col,
            canonicalize_smiles_for_checks=True
        )
        last_stats = stats

        try:
            assert_blind_start_ok(stats, min_test_samples=min_test_samples)
            if verbose:
                print(f"[blind_start] OK with seed={cur_seed} | "
                      f"train/valid/test={stats['n_train']}/{stats['n_valid']}/{stats['n_test']} "
                      f"(test={stats['frac_test']:.3f})")
            return train_idx, valid_idx, test_idx, stats
        except AssertionError as e:
            if verbose:
                print(f"[blind_start] retry {k+1}/{max_tries} seed={cur_seed} failed: {e}")

    raise RuntimeError(f"Failed to generate a valid blind start split after {max_tries} tries. "
                       f"Last stats: {last_stats}")


def assert_partition_ok(df, train_idx, valid_idx, test_idx):
    """
    Verify that train/valid/test form a complete, non-overlapping partition of df.
    
    Args:
        df: Original DataFrame
        train_idx, valid_idx, test_idx: Split indices
        
    Raises:
        AssertionError: If indices overlap or don't cover all rows
    """
    all_idx = set(df.index.tolist())
    a, b, c = set(train_idx), set(valid_idx), set(test_idx)
    
    overlap_ab = len(a & b)
    overlap_ac = len(a & c)
    overlap_bc = len(b & c)
    
    assert overlap_ab == 0, f"Train-valid overlap: {overlap_ab} indices"
    assert overlap_ac == 0, f"Train-test overlap: {overlap_ac} indices"
    assert overlap_bc == 0, f"Valid-test overlap: {overlap_bc} indices"
    
    union = a | b | c
    missing = all_idx - union
    extra = union - all_idx
    
    assert len(missing) == 0, f"Missing {len(missing)} rows from partition"
    assert len(extra) == 0, f"Extra {len(extra)} indices not in original df"
    
    return True


def save_split_indices(dataset_name, split_name, train_idx, valid_idx, test_idx):
    """
    Save split indices to files.
    
    Args:
        dataset_name: Name of the dataset
        split_name: Name of the split (e.g., 'scaffold', 'cold_protein')
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    save_dir = f'./dataset/data/DTI/processed/{dataset_name}/split_indices'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f'{save_dir}/{split_name}_train_indices.npy', np.array(train_idx))
    np.save(f'{save_dir}/{split_name}_valid_indices.npy', np.array(valid_idx))
    np.save(f'{save_dir}/{split_name}_test_indices.npy', np.array(test_idx))
    
    print(f"Saved {split_name} split indices for {dataset_name} dataset")
    print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")


def load_split_indices(dataset_name, split_name):
    """
    Load split indices from files.
    
    Args:
        dataset_name: Name of the dataset
        split_name: Name of the split (e.g., 'scaffold', 'cold_protein')
        
    Returns:
        train_idx, valid_idx, test_idx: Indices for train, validation and test sets
    """
    save_dir = f'./dataset/data/DTI/processed/{dataset_name}/split_indices'
    
    train_idx = np.load(f'{save_dir}/{split_name}_train_indices.npy').tolist()
    valid_idx = np.load(f'{save_dir}/{split_name}_valid_indices.npy').tolist()
    test_idx = np.load(f'{save_dir}/{split_name}_test_indices.npy').tolist()
    
    print(f"Loaded {split_name} split indices for {dataset_name} dataset")
    print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    
    return train_idx, valid_idx, test_idx