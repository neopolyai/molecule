import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Batch
from torch_geometric.data import Data
from chemutils import get_mol, get_clique_mol

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



class MoleculeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split(',') for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx][0]
        mol_graph = MolGraph(smiles) 
        return mol_graph



class MolGraph(object):
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # # TODO: Encode partial charges into mol_graph!
        # partial_charges = get_gasteiger_partial_charges(self.mol)

        # Retrieve atom features/types
        atom_features_list = []        
        for atom in self.mol.GetAtoms():
            atom_feature = [
                features['atomic_num'].index(atom.GetAtomicNum())] + [
                    features['degree'].index(atom.GetDegree())] + [
                        features['chirality'].index(atom.GetChiralTag())] + [
                            features['formal_charge'].index(atom.GetFormalCharge())]
            atom_features_list.append(atom_feature)

        # Retrieve edge indices and types
        edges_list = []
        edge_features_list = []
        for bond in self.mol.GetBonds():
            # Update edge index list
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges_list.append((i, j))
            edges_list.append((j, i))
            # Update edge feature list
            edge_feature = [
                features['bonds'].index(bond.GetBondType())] + [
                    features['bond_inring'].index(bond.IsInRing())] + [
                        features['stereo'].index(bond.GetStereo())] + [
                            features['bond_isconjugated'].index(bond.GetIsConjugated())]
            edge_features_list.append(edge_feature)
            edge_features_list.append(edge_feature)

        # Assign atom features, edge indices, and edge attributes to molecule object
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list), dtype=torch.long) 

        # Get num_atoms and num_motifs
        num_atoms = self.x_nosuper.size(0)  
        cliques = motif_decomp(self.mol, self.smiles)
        num_motif = len(cliques)

        # NOTE: We need to encode more details in motif_x 
        if num_motif > 0:
            self.num_part = (num_atoms, num_motif, 1)

            # Update self.x  with motif node attributes
            super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(self.x_nosuper.device)
            motif_x = torch.tensor([[MAX_ATOM_TYPE+1, 0, 0, 3]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)

            # Create super_edge_index and super_edge_attr (self-loop)
            super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            super_edge_attr = torch.zeros(num_motif, 4)
            super_edge_attr[:,0] = 5               
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            # Update self.edge_index with motif edge indices
            motif_edge_index, frag_edge_index = [], []
            for k, motif in enumerate(cliques):
                motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
                frag_edge_index = frag_edge_index + [[i, k] for i in motif]     
            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.frags = torch.tensor(np.array(frag_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

            # Update self.edge_attr with motif edge attributes
            motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 4)
            motif_edge_attr[:,0] = 6 
            motif_edge_attr = motif_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim=0)
       
        else:
            print("Number of motifs is 0!")
            self.num_part = (num_atoms, 0, 1)            
            
            # Update x with self-loop
            super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)
            # Add self-loop edge index
            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)
            # Add self-loop edge attribute
            super_edge_attr = torch.zeros(num_atoms, 4)
            super_edge_attr[:,0] = 5 
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim=0)



    def size_node(self):
        return self.x.size()[0]
    def size_edge(self):
        return self.edge_attr.size()[0]
    def size_atom(self):
        return self.x_nosuper.size()[0]
    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]
    def size_motif(self):
        return self.num_part[1]




# Helper function to convery molgraph to graph
def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)

    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch



# Helper function to get motif decomposition
def motif_decomp(mol, smiles):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    # Fragment molecule into motifs and replace BRICS bonds
    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 
    # else:
    #     # NOTE: Identify all molecules that are not decomposable
    #     with open('all_notdecomposable.txt', 'a') as f:
    #         f.write(smiles + "\n")

    # Merge cliques to group atoms by fragments they belong to
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
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    # Get smallest set of smallest rings from motif cliques
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
                    cliques[i] = list(set(cliques[i]) - set(list(ring)))
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    return cliques