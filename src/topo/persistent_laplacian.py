import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import copy
import time
import os
import sys




def atmtyp_to_ele( st ):
    # C,N,O,S,P,H,F,Cl,Br,I
    st = st.strip()
    if len(st) == 1:
        return st
    elif st[0] == 'H':
        return 'H'
    elif st[0] == 'N':
        return 'N'
    elif st[0] == 'O':
        return 'O'
    elif st[0] == 'S':
        return 'S'
    elif st[0] == 'P':
        return 'P'
    elif st[0] == 'C' and st[0:2] not in ['Cl','CL']:
        return 'C'
    elif st[0] == 'F':
        return 'F'
    elif st[0] == 'I':
        return 'I'
    elif st[0:2] in ['Cl', 'CL','cl']:
        return 'CL'
    elif st[0:2] in ['BR','Br','br']:
        return 'BR'
    elif st[1] in ['H']:
        return 'H'
    else:
        print(st, 'Not in dictionary')
        return
residue_to_one_letter = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

class Atom:
    def __init__(self,atype,resname,chain,resid,coord,charge=None):
        self.AType = atype
        self.Coord = coord
        self.ResName = resname
        self.ResId = resid
        self.Chain = chain
        self.charge = charge

class ProteinLigand:
    def __init__(self,pdb,typ,pdb_folder=None,pdb_feature_folder=None,pqr_folder=None,pqr_feature_folder=None):
        self.pdb= pdb
        self.pdb_folder = pdb_folder
        self.pdb_feature_folder = pdb_feature_folder
        self.pqr_folder = pqr_folder
        self.pqr_feature_folder = pqr_feature_folder
        self.Protein_Atoms = []
        self.Protein_AtomCoord = []
        self.Protein_AtomCharge = []
        self.Ligand_Atoms = []
        self.Ligand_AtomCoord = []
        self.Ligand_AtomCharge = []
        self.P_Index_List = []
        self.L_Index_List = []
        
        if typ=='euclidean':
            self.get_euclidean_pl()
        elif typ=='charge':
            self.get_charge_pl()
        
        
    
    def read_atom_from_pdb(self):
        # protein
        filename1 = self.pdb_folder + '/'+self.pdb+'/'+self.pdb+'_pocket.pdb'
        filename2 = self.pdb_folder + '/'+self.pdb+'/'+self.pdb+'_ligand.mol2'
        
        f = open(filename1)
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                atom = Atom(atype=atmtyp_to_ele(line[12:16]), resname=residue_to_one_letter[line[17:20]], chain=line[21], resid=int(line[22:26]),
                            coord=[float(line[30:38]),float(line[38:46]),float(line[46:54])],
                             )
                self.Protein_Atoms.append(atom)
                self.Protein_AtomCoord.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        self.Protein_AtomCoord = np.array(self.Protein_AtomCoord)
        
        
        # ligand
        f = open(filename2)
        contents = f.readlines()
        f.close()
        
        start = 0
        end = 0
        for jj in range(len(contents)):
            if contents[jj][0:13]=='@<TRIPOS>ATOM':
                start = jj + 1
                continue
            if contents[jj][0:13]=='@<TRIPOS>BOND':
                end = jj - 1
                break
        for kk in range(start,end+1):
            if contents[kk][8:17]=='thiophene':
                print('thiophene',kk)
            line = contents[kk]
            atom = Atom(atype=atmtyp_to_ele(line[47]), resname='ligand', chain='ligand', resid='ligand',
                            coord=[float(line[16:26]),float(line[26:36]),float(line[36:46])],
                             )
            self.Ligand_Atoms.append(atom)
            self.Ligand_AtomCoord.append([float(line[16:26]),float(line[26:36]),float(line[36:46])])
        self.Ligand_AtomCoord = np.array(self.Ligand_AtomCoord)
    
    def read_atom_from_pqr(self):
        # protein
        filename1 = self.pqr_folder+'/'+self.pdb+'.pqr'
        filename2 = self.pdb_folder+'/'+self.pdb+'/'+self.pdb+'_ligand.mol2'
        
        f = open(filename1)
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                atom = Atom(atype=atmtyp_to_ele(line[12:16]), resname=residue_to_one_letter[line[17:20]], chain=line[21], resid=int(line[22:26]),
                            coord=[float(line[30:38]),float(line[38:46]),float(line[46:54])], charge=float(line[54:62])
                             )
                self.Protein_Atoms.append(atom)
                self.Protein_AtomCoord.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                self.Protein_AtomCharge.append(float(line[54:62]))
        self.Protein_AtomCoord = np.array(self.Protein_AtomCoord)
        self.Protein_AtomCharge = np.array(self.Protein_AtomCharge)
        
        # ligand
        f = open(filename2)
        contents = f.readlines()
        f.close()
        
        start = 0
        end = 0
        for jj in range(len(contents)):
            if contents[jj][0:13]=='@<TRIPOS>ATOM':
                start = jj + 1
                continue
            if contents[jj][0:13]=='@<TRIPOS>BOND':
                end = jj - 1
                break
        for kk in range(start,end+1):
            if contents[kk][8:17]=='thiophene':
                print('thiophene',kk)
            line = contents[kk]
            atom = Atom(atype=atmtyp_to_ele(line[47]), resname='ligand', chain='ligand', resid='ligand',
                            coord=[float(line[16:26]),float(line[26:36]),float(line[36:46])],charge=float(line[70:76])
                             )
            self.Ligand_Atoms.append(atom)
            self.Ligand_AtomCoord.append([float(line[16:26]),float(line[26:36]),float(line[36:46])])
            self.Ligand_AtomCharge.append(float(line[70:76]))
        self.Ligand_AtomCoord = np.array(self.Ligand_AtomCoord)
        self.Ligand_AtomCharge = np.array(self.Ligand_AtomCharge)
    
        
    def set_euclidean_index_list(self):
        ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'P':4}
        
        # protein C, N, O, S
        self.P_Index_List = [ [] for _ in range(4) ]
        for idx in range(len(self.Protein_Atoms)):
            atom = self.Protein_Atoms[idx]
            if atom.AType in ele2index:
                self.P_Index_List[ele2index[atom.AType]].append(idx)
            
        
        # ligand C, N, O, S, P
        self.L_Index_List = [ [] for _ in range(5) ]
        for idx in range(len(self.Ligand_Atoms)):
            atom = self.Ligand_Atoms[idx]
            if atom.AType in ele2index:
                self.L_Index_List[ele2index[atom.AType]].append(idx)
    
    def set_charge_index_list(self):
        ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'H':4, 'P':5}
        
        # protein C, N, O, S, H
        self.P_Index_List = [ [] for _ in range(5) ]
        for idx in range(len(self.Protein_Atoms)):
            atom = self.Protein_Atoms[idx]
            if atom.AType in ele2index:
                self.P_Index_List[ele2index[atom.AType]].append(idx)
            
        
        # ligand C, N, O, S, H, P
        self.L_Index_List = [ [] for _ in range(6) ]
        for idx in range(len(self.Ligand_Atoms)):
            atom = self.Ligand_Atoms[idx]
            if atom.AType in ele2index:
                self.L_Index_List[ele2index[atom.AType]].append(idx)
    
    def set_cate_index_list(self):
        cate = [ ['G', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W'], ['S', 'T', 'N', 'Q', 'Y', 'C'], ['D','E'], ['K','R','H'] ]
        cate2index = {}
        for i in range(len(cate)):
            for res in cate[i]:
                cate2index[res] = i
        
        # protein 4 category
        self.P_Index_List = [ [] for _ in range(4) ]
        for idx in range(len(self.Protein_Atoms)):
            atom = self.Protein_Atoms[idx]
            if atom.ResName in cate2index:
                self.P_Index_List[cate2index[atom.ResName]].append(idx)
        
        
        ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'P':4}
        # ligand C, N, O, S, P
        self.L_Index_List = [ [] for _ in range(5) ]
        for idx in range(len(self.Ligand_Atoms)):
            atom = self.Ligand_Atoms[idx]
            if atom.AType in ele2index:
                self.L_Index_List[ele2index[atom.AType]].append(idx)
    
    def set_index_list_more(self):
        # protein C, N, O, S, CN, CO, CS, NO, NS, OS, CNOS
        self.P_Index_List = [ [] for _ in range(11) ]
        for idx in range(len(self.Protein_Atoms)):
            atom = self.Protein_Atoms[idx]
            if atom.AType=='C':
                for k in [0,4,5,6,10]:
                    self.P_Index_List[k].append(idx)
            elif atom.AType=='N':
                for k in [1,4,7,8,10]:
                    self.P_Index_List[k].append(idx)
            elif atom.AType=='O':
                for k in [2,5,7,9,10]:
                    self.P_Index_List[k].append(idx)
            elif atom.AType=='S':
                for k in [3,6,8,9,10]:
                    self.P_Index_List[k].append(idx)
        
        # ligand C, N, O, S, CN, CO, CS, NO, NS, OS, NP, F-CL-BR-I, all
        self.L_Index_List = [ [] for _ in range(13) ]
        for idx in range(len(self.Ligand_Atoms)):
            atom = self.Ligand_Atoms[idx]
            if atom.AType=='C':
                for k in [0,4,5,6,12]:
                    self.L_Index_List[k].append(idx)
            elif atom.AType=='N':
                for k in [1,4,7,8,10,12]:
                    self.L_Index_List[k].append(idx)
            elif atom.AType=='O':
                for k in [2,5,7,9,12]:
                    self.L_Index_List[k].append(idx)
            elif atom.AType=='S':
                for k in [3,6,8,9,12]:
                    self.L_Index_List[k].append(idx)
            elif atom.AType=='P':
                for k in [10,12]:
                    self.L_Index_List[k].append(idx)
            elif atom.AType in ['F','CL','BR','I']:
                self.L_Index_List[11].append(idx)
    
    def get_euclidean_L0(self):
        def get_statistic(value):
            if len(value)==0:
                return np.array([0 for _ in range(7)])
            else:
                return np.array([ np.max(value),np.min(value),np.sum(value),np.mean(value),np.std(value),np.var(value),len(value) ])
                
        fil = [ round(0.1*i,1) for i in range(1,101) ]
        fea = np.zeros((len(fil),20,8))
    
        for i in range(4):
            index1 = self.P_Index_List[i]
            atom1 = self.Protein_AtomCoord[index1]
            for j in range(5):
                index2 = self.L_Index_List[j]
                atom2 = self.Ligand_AtomCoord[index2]
                m,n = atom1.shape[0],atom2.shape[0]
                block_dis = cdist(atom1, atom2, metric='euclidean')
                dis_m = np.full((m + n, m + n), 99.0)
                dis_m[:m,m:] = block_dis
                dis_m[m:,:m] = block_dis.T
                
                now = i*5+j
                for k in range(len(fil)):
                    now_fil = fil[k]
                    L = copy.deepcopy(dis_m)
                    L[L<=now_fil] = -1
                    L[L>0] = 0
                    for s in range(m+n):
                        L[s,s] = - np.sum(L[s,:])
                    eig = np.linalg.eigvalsh(L)
                    fea[k,now,0] = len(eig[eig<1e-8])
                    eig = eig[eig>1e-8]
                    fea[k,now,1:8] = get_statistic(eig)
                    
        if not os.path.isdir(self.pdb_feature_folder):
            os.makedirs(self.pdb_feature_folder)
        np.save(self.pdb_feature_folder + '/' + self.pdb + '.npy',fea)
    
    def get_charge_L0(self,eps = 1e-12):
        # zeta function
        def get_spectral_moment(ls,k):
            res = 0
            for i in range(len(ls)):
                if ls[i]!=0:
                    res = res + pow(ls[i],k)
            return res
    
        def get_zeta_function(value):
            if len(value)==0:
                return np.array([0 for _ in range(11)])
            else:
                res = []
                for k in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                    res.append(get_spectral_moment(value,k))
                return np.array(res)
                
        fil = [ round(0.01*i,1) for i in range(1,101) ]
        fea = np.zeros((len(fil),30,11))
        
        for i in range(5):
            index1 = self.P_Index_List[i]
            atom1 = self.Protein_AtomCoord[index1]
            charge1 = self.Protein_AtomCharge[index1]
            for j in range(6):
                index2 = self.L_Index_List[j]
                atom2 = self.Ligand_AtomCoord[index2]
                charge2 = self.Ligand_AtomCharge[index2]
                
                m, n = atom1.shape[0], atom2.shape[0]
                dis = cdist(atom1, atom2, metric="euclidean")
                d_safe = np.maximum(dis, eps)
                
                qq = 100*np.outer(charge1, charge2)
                block_dis = 1.0 / (1.0 + np.exp(-(qq / d_safe)))
                
                dis_m = np.full((m + n, m + n), 99.0)
                dis_m[:m, m:] = block_dis
                dis_m[m:, :m] = block_dis.T
                
                now = i*6+j
                for k in range(len(fil)):
                    now_fil = fil[k]
                    L = copy.deepcopy(dis_m)
                    L[L<=now_fil] = -1
                    L[L>0] = 0
                    for s in range(m+n):
                        L[s,s] = - np.sum(L[s,:])
                    eig = np.linalg.eigvalsh(L)
                    eig = eig[eig>1e-8]
                    fea[k,now] = get_zeta_function(eig)
                    
        if not os.path.isdir(self.pqr_feature_folder):
            os.makedirs(self.pqr_feature_folder)
        np.save(self.pqr_feature_folder+'/'+self.pdb+'.npy',fea)
    
    def get_euclidean_pl(self):
        self.read_atom_from_pdb()
        self.set_euclidean_index_list()
        self.get_euclidean_L0()
    
    def get_charge_pl(self):
        self.read_atom_from_pqr()
        self.set_charge_index_list()
        self.get_charge_L0()
    
        
    
        
