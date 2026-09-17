from .topo import ProteinLigand




def prepare_laplacian(dataname,para):
    bad = ['3tei',  '4as6']
    if dataname in [2007,2013,2016]:
        f = open('./data/train_data_' + str(dataname) + '.txt')
        train_pdb = eval(f.read())
        train_pdb = [p for p in train_pdb if p not in bad]
        f.close()
        
        f = open('./data/test_data_' + str(dataname) + '.txt')
        test_pdb = eval(f.read())
        test_pdb = [p for p in test_pdb if p not in bad]
        f.close()
        data = train_pdb+test_pdb
    elif dataname in [2020]:
        filename = './data/pretrain-pdbs.txt'
        f = open(filename)
        data = eval(f.read())
        pdbs = [p for p in data if p not in miss]
        f.close()
    
    print(f'Prepare feature for PDBbind-v{dataname}, all {len(data)} samples ')
    
    
    
    for i in range(len(data)):
        pdb = data[i]
        tmp = ProteinLigand(pdb,'euclidean',pdb_folder=para.pdb_folder,pdb_feature_folder=para.pdb_feature_folder)
        if dataname in [2007,2013,2016]:
            tmp = ProteinLigand(pdb,'charge',pdb_folder=para.pdb_folder,pqr_folder=para.pqr_folder,pqr_feature_folder=para.pqr_feature_folder)
        print(f'{pdb} ok, {i+1}/{len(data)}')



