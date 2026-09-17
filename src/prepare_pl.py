from .topo import ProteinLigand
from ._runtime_guard import read_literal_sequence, require_choice




def prepare_laplacian(dataname,para):
    dataname = require_choice(dataname, (2007, 2013, 2016, 2020), 'feature dataset year')
    bad = ['3tei',  '4as6']
    if dataname in [2007,2013,2016]:
        train_pdb = read_literal_sequence(
            './data/train_data_' + str(dataname) + '.txt', excluded=bad
        )
        test_pdb = read_literal_sequence(
            './data/test_data_' + str(dataname) + '.txt', excluded=bad
        )
        data = train_pdb+test_pdb
    elif dataname in [2020]:
        filename = './data/pretrain-pdbs.txt'
        data = read_literal_sequence(filename)
    
    print(f'Prepare feature for PDBbind-v{dataname}, all {len(data)} samples ')
    
    
    
    for i in range(len(data)):
        pdb = data[i]
        tmp = ProteinLigand(pdb,'euclidean',pdb_folder=para.pdb_folder,pdb_feature_folder=para.pdb_feature_folder)
        if dataname in [2007,2013,2016]:
            tmp = ProteinLigand(pdb,'charge',pdb_folder=para.pdb_folder,pqr_folder=para.pqr_folder,pqr_feature_folder=para.pqr_feature_folder)
        print(f'{pdb} ok, {i+1}/{len(data)}')



