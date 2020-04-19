import numpy as np
from rdkit import Chem
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def get_molecules(filename) :
    if os.path.exists('data/data.dataset') :
        with open('data/data.dataset', 'rb') as f :
            data = pickle.load(f)
            return data['A'], data['X']
    data = list(filter(lambda x: x is not None and x.GetNumAtoms() > 1, Chem.SDMolSupplier(filename)))
    max_atom_nb = max(mol.GetNumAtoms() for mol in data)
    As = np.zeros((len(data), max_atom_nb, max_atom_nb), dtype=int)
    Xs = np.zeros((len(data), max_atom_nb), dtype=int)
    for i, mol in enumerate(data) :
        n = mol.GetNumAtoms()
        bonds = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()] for bond in mol.GetBonds()]).reshape(-1, 3)
        As[i][bonds[:, 0], bonds[:, 1]] = bonds[:, 2]
        As[i][bonds[:, 1], bonds[:, 0]] = bonds[:, 2]
        Xs[i][:n] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    with open('data/data.dataset', 'wb') as f : 
        pickle.dump({'A' : As, 'X' : Xs}, f)
    return As, Xs


def matrices2mol(node_labels, edge_labels, strict=False):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(int(node_label)))
    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), Chem.BondType.values[edge_labels[start, end]])
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None
    return mol

def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))

