import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout, device, verbose, singleT = True, simpleO = True):
        
        super(Fingerprint, self).__init__()
        
        self.verbose = verbose
        # graph attention for atom embedding
        print('input_feature_dim',input_feature_dim)
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim).to(device)
        
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim).to(device)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim).to(device) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1).to(device) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim).to(device) for r in range(radius)])
        # graph attention for molecule embedding single or multiple layer
        if singleT:
            self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim).to(device)
            self.mol_align = nn.Linear(2*fingerprint_dim,1).to(device)
            self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim).to(device)
        else:
            self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
            self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
            self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])

        self.dropout = nn.Dropout(p=p_dropout).to(device)
        
        if simpleO:
            self.output = nn.Linear(fingerprint_dim, output_units_num).to(device)
        else:
            self.output = nn.Sequential(
            nn.Linear(fingerprint_dim, 100),
            nn.LayerNorm(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(100, output_units_num)
        ).to(device)
        
        self.device = device
        self.radius = radius
        self.T = T
        self.verbose = verbose
        self.singleT = singleT


    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        
        if self.verbose:
            print('atom_list',atom_list)
            print('bond_list',bond_list)
            print('atom_mask',atom_mask)
            print('bond_degree_list',bond_degree_list)
            print('atom_degree_list',atom_degree_list)
            print('atom_mask',atom_mask)

        atom_mask = atom_mask.unsqueeze(2)
        if self.verbose:
            print('atom_mask unsqueeze',atom_mask.shape)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        if self.verbose:
            print('batch_size',batch_size)
            print('mol_length',mol_length)
            print('num_atom_feat',num_atom_feat)

        atom_feature = F.leaky_relu(self.atom_fc(atom_list.to(self.device)))
        if self.verbose:
            print('atom_feature',atom_feature.shape)
            print('atom_feature',atom_feature)

        self.verbose = False
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0).to(self.device)
        if self.verbose:
            print('bond_neighbor',bond_neighbor.shape)
            print('bond_neighbor',bond_neighbor)

        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0).to(self.device)
        # then concatenate them
        if self.verbose:
            print('atom_neighbor',atom_neighbor.shape)
            print('atom_neighbor',atom_neighbor)
        
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        if self.verbose:
            print('neighbor_feature',neighbor_feature.shape)
            print('neighbor_feature',neighbor_feature)

        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))
        if self.verbose:
            print('neighbor_feature projection',neighbor_feature.shape)
            print('neighbor_feature',neighbor_feature)

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone().to(self.device)
        if self.verbose:
            print('attend_mask',attend_mask.shape)
            print('attend_mask',attend_mask)

        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.FloatTensor).unsqueeze(-1).to(self.device)
        if self.verbose:
            print('attend_mask unsqueeze',attend_mask.shape)
            print('attend_mask unsqueeze',attend_mask)

        softmax_mask = atom_degree_list.clone().to(self.device)
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        if self.verbose:
            print('softmax_mask',softmax_mask.shape)
        
        softmax_mask = softmax_mask.type(torch.FloatTensor).unsqueeze(-1).to(self.device)
        if self.verbose:
            print('softmax_mask unsqueeze',softmax_mask.shape)
            print('softmax_mask',softmax_mask)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim).to(self.device)
        if self.verbose:
            print('mol_length',mol_length)
            print('max_neighbor_num',max_neighbor_num)
            print('batch_size',batch_size)
            print('atom_feature_expand',atom_feature_expand.shape)
            print('atom_feature_expand',atom_feature_expand)
       
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        if self.verbose:
            print('feature_align',feature_align.shape)
            print('feature_align',feature_align)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align))).to(self.device)
        if self.verbose:
            print('align_score',align_score.shape)
            print('align_score',align_score)

        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
        if self.verbose:
            print('attention_weight',attention_weight.shape)
            print('attention_weight',attention_weight)

        attention_weight = attention_weight * attend_mask
        if self.verbose:
            print('attention_weight * mask',attention_weight.shape)
            print('attention_weight * mask',attention_weight)

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        if self.verbose:
            print('neighbor_feature_transform',neighbor_feature_transform.shape)
            print('neighbor_feature_transform',neighbor_feature_transform)

        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
        if self.verbose:
            print('context',context.shape)
            print('context',context)
            
            
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        if self.verbose:
            print('context_reshape',context_reshape.shape)
            print('context_reshape',context_reshape)

        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        if self.verbose:
            print('atom_feature_reshape',atom_feature_reshape.shape)
            print('atom_feature_reshape',atom_feature_reshape)
        
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        if self.verbose:
            print('atom_feature_reshape after gru',atom_feature_reshape.shape)
            print('atom_feature_reshape after gru',atom_feature_reshape)

        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
        if self.verbose:
            print('atom_feature after gru 0:',atom_feature.shape)
            print('atom_feature after gru 0:',atom_feature)

        if self.verbose:
            print("radius",self.radius)
        #do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            if self.verbose:
                print('d:',d)
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
            if self.verbose:
                print('align_score',align_score.shape)
                print('align_score',align_score)
                
                

            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
            if self.verbose:
                print('attention_weight',attention_weight.shape)
                print('attention_weight',attention_weight)
            attention_weight = attention_weight * attend_mask

            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
            if self.verbose:
                print('neighbor_feature_transform',neighbor_feature_transform.shape)
                print('neighbor_feature_transform',neighbor_feature_transform)

            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
            if self.verbose:
                print('context',context.shape)
                print('context',context)

            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
            if self.verbose:
                print('context_reshape',context_reshape.shape)
                print('context_reshape',context_reshape)

            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            if self.verbose:
                print('atom_feature after gru',atom_feature_reshape.shape)
                print('atom_feature after gru',atom_feature_reshape)

            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            if self.verbose:
                print('atom_feature after gru and reshape',atom_feature.shape)
                print('atom_feature after gru and reshape',atom_feature)

            # do nonlinearity
            activated_features = F.relu(atom_feature)
            if self.verbose:
                print('activated_features',activated_features.shape)
                print('activated_features',activated_features)

        if self.verbose:
            print('*'*10)
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        if self.verbose:
            print('mol_feature',mol_feature.shape)
            print('mol_feature',mol_feature)
        
        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)           
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.FloatTensor).to(self.device)
        if self.verbose:
            print('mol_softmax_mask',mol_softmax_mask.shape)
            print('mol_softmax_mask',mol_softmax_mask)

        for t in range(self.T):
            if self.verbose:
                print('-'*t*10)
                print('T:',t)
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            if self.verbose:
                print('mol_prediction_expand',mol_prediction_expand.shape)
                print('mol_prediction_expand',mol_prediction_expand)

            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            if self.verbose:
                print('mol_align',mol_align.shape)
                print('mol_align',mol_align)
            if self.singleT:
                mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            else:
                mol_align_score = F.leaky_relu(self.mol_align[t](mol_align))
                
            
            if self.verbose:
                print('mol_align_score',mol_align_score.shape)
                print('mol_align_score',mol_align_score)

            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            if self.verbose:
                print('mol_attention_weight',mol_attention_weight.shape)
                print('mol_attention_weight',mol_attention_weight)

            mol_attention_weight = mol_attention_weight * atom_mask
            if self.singleT:

                activated_features_transform = self.mol_attend(self.dropout(activated_features))
            else:
                activated_features_transform = self.mol_attend[t](self.dropout(activated_features))
                
                    
                
            if self.verbose:
                print('activated_features_transform',activated_features_transform.shape)
                print('activated_features_transform',activated_features_transform)

#           aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
            if self.verbose:
                print('mol_context',mol_context.shape)
                print('mol_context',mol_context)

            mol_context = F.elu(mol_context)
            if self.singleT:
                mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            else:
                mol_feature = self.mol_GRUCell[t](mol_context, mol_feature)
                
            if self.verbose:
                print('mol_feature',mol_feature.shape)
                print('mol_feature',mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)           
            if self.verbose:
                print('activated_features_mol',activated_features_mol.shape)
                print('activated_features_mol',activated_features_mol)
           
        mol_prediction = self.output(self.dropout(mol_feature))
            
        return atom_feature, mol_prediction