import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from chem_utils import get_mol, get_clique_mol

"""
NOTE: Polymer SMILES representation
Polymer bonds * are represented as atoms "0"
() in (*) is represented as single-bond branches

'hybridization' : [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP, 
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, 
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
    ]
"""

features = {
    'atomic_num': [0,1,5,6,7,8,9,11,14,15,16,17,19,20,26,27,28,30,35,53],
    'degree' : [0,1,2,3,4,5,6],
    'formal_charge' : [-3,-2,-1,0,1,2,3],
    'chirality' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER],
    'bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC],
    'bond_inring': [
        None, 
        False, 
        True],
    'bond_isconjugated': [
        None,
        False,
        True],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOZ]
}
MAX_ATOM_TYPE = len(features['atomic_num'])



def get_gasteiger_partial_charges(mol, n_iter=12):
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter, throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
    return partial_charges


    
def mol_to_graph_data_obj_simple(mol, smiles):
    # # TODO: Encode partial charges into mol_graph!
    # partial_charges = get_gasteiger_partial_charges(mol)

    # Get atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            features['atomic_num'].index(atom.GetAtomicNum())] + [
                features['degree'].index(atom.GetDegree())] + [
                    features['chirality'].index(atom.GetChiralTag())] + [
                        features['formal_charge'].index(atom.GetFormalCharge())]
        atom_features_list.append(atom_feature)

    # Get edge indices and edge features
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        # Append start and end atoms of the bond
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_list.append((i, j))
        edges_list.append((j, i))
        # Append features/attributes of the bond
        edge_feature = [
            features['bonds'].index(bond.GetBondType())] + [
                features['bond_inring'].index(bond.IsInRing())] + [
                    features['stereo'].index(bond.GetStereo())] + [
                        features['bond_isconjugated'].index(bond.GetIsConjugated())]
        edge_features_list.append(edge_feature)
        edge_features_list.append(edge_feature)
    
    # Tesorize features and indices
    x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr_nosuper = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    # Get num_atoms and num_motifs
    num_atoms = x_nosuper.size(0)    
    cliques = motif_decomp(mol, smiles)
    num_motif = len(cliques)

    # NOTE: We need to encode more details in motif_x 
    if num_motif > 0:
        # Update self.x 
        super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(x_nosuper.device)
        motif_x = torch.tensor([[MAX_ATOM_TYPE+1, 0, 0, 3]]).repeat_interleave(num_motif, dim=0).to(x_nosuper.device)
        x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

        # Create super_edge_index and super_edge_attr (self-loop)
        super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        super_edge_attr = torch.zeros(num_motif, 4)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)

        # Update self.edge_index with motif edge indices
        motif_edge_index = []
        for k, motif in enumerate(cliques):
            motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

        # Update self.edge_index with motif edge attributes
        motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 4)
        motif_edge_attr[:,0] = 6 
        motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

    else:
        # print('Number of motifs is 0!')
        # Update x with self-loop
        super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(x_nosuper.device)
        x = torch.cat((x_nosuper, super_x), dim=0)
        # Add self-loop edge index
        super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)
        # Add self-loop edge attribute
        super_edge_attr = torch.zeros(num_atoms, 4)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim = 0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



def motif_decomp(mol, smiles):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]

    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1: 
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i]=[]
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]
    return cliques



# Helper function to load in dataset in CSV format
def _load_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    
    # Convert homopolymer SMILES to mol object "x" for encoder
    homo_list = input_df['smiles']
    homo_mol_objs = [AllChem.MolFromSmiles(s) for s in homo_list]

    # Convert copolymer SMILES to mol object "x" for encoder
    copo_list = input_df['copo_smiles']
    copo_mol_objs = []
    for s in copo_list:
        if type(s) == str:
            copo_mol_objs.append(AllChem.MolFromSmiles(s))
        else:
            copo_mol_objs.append(None)
 
    # Get "y" for gnn_graphpred_linear and rewards for bayesian network
    tasks = ['HOMO', 'LUMO', 'electrochemical_gap', 'optical_gap', 'PCE', 'V_OC', 'J_SC', 'fill_factor']
    labels = input_df[tasks]
    labels = (labels.replace(0, -1)).fillna(0)   
    
    assert len(homo_list) == len(labels)
    assert len(homo_list) == len(homo_mol_objs)
    assert len(homo_mol_objs) == len(copo_mol_objs)
    return homo_list, homo_mol_objs, copo_list, copo_mol_objs, labels.values



class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        homo_smiles_list, copo_smiles_list, data_list = [], [], []
        # Load in dataset in CSV format
        raw_path = '../data/training/raw/hopv.csv'
        homo_smiles, homo_mol_objs, copo_smiles, copo_mol_objs, labels = _load_dataset(raw_path)
        for i in range(len(homo_smiles)):
            # Convert homopolymer to mol object
            homo_smiles_list.append(homo_smiles[i])
            homo_mol = homo_mol_objs[i]
            data = mol_to_graph_data_obj_simple(homo_mol, homo_smiles[i])

            # Convert copolymer to mol object
            copo_smiles_list.append(copo_smiles[i])
            copo_mol = copo_mol_objs[i]
            if copo_mol != None:
                copo_data = mol_to_graph_data_obj_simple(copo_mol, copo_smiles[i])
            else:
                copo_data = Data(x=None, edge_index=None, edge_attr=None)

            # Add atributes to the mol object data
            data.copo_x = copo_data.x
            data.copo_edge_index = copo_data.edge_index
            data.copo_edge_attr = copo_data.edge_attr
            data.id = torch.tensor([i]) 
            data.y = torch.tensor(labels[i, :])
            data_list.append(data)

        # Write homo_smiles_list and copo_smiles_list in processed paths
        homo_smiles_series = pd.Series(homo_smiles_list)
        homo_smiles_series.to_csv(os.path.join(self.processed_dir, 'homo_smiles.csv'), index=False, header=False)
        copo_smiles_series = pd.Series(copo_smiles_list)
        copo_smiles_series.to_csv(os.path.join(self.processed_dir, 'copo_smiles.csv'), index=False, header=False)

        # Prefilter and pretransform data
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save geometric processed data in processed path
        homo_data, homo_slices = self.collate(data_list)
        torch.save((homo_data, homo_slices), self.processed_paths[0])



# Run this to create the finetune dataset
if __name__ == "__main__":
    root = "../data/training"
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = MoleculeDataset(root)