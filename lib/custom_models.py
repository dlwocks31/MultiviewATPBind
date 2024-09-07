from collections import OrderedDict
from torch import Tensor
from transformers import BertModel, BertTokenizer, AutoTokenizer, EsmModel
import torch
from torchdrug import core, models
from gvp.models import GVPConvLayer, GVP, LayerNorm
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def separate_alphabets(text):
    separated_text = ""
    for char in text:
        if char.isalpha():
            separated_text += char + " "
    return separated_text.strip()


lm_type_map = {
    'bert': (BertModel, BertTokenizer, "Rostlab/prot_bert", 30),
    'esm-t6': (EsmModel, AutoTokenizer, "facebook/esm2_t6_8M_UR50D", 6),
    'esm-t12': (EsmModel, AutoTokenizer, "facebook/esm2_t12_35M_UR50D", 12),
    'esm-t30': (EsmModel, AutoTokenizer, "facebook/esm2_t30_150M_UR50D", 30),
    'esm-t33': (EsmModel, AutoTokenizer, "facebook/esm2_t33_650M_UR50D", 33),
    'esm-t36': (EsmModel, AutoTokenizer, "facebook/esm2_t36_3B_UR50D", 36),
}

class LMGearNetModel(torch.nn.Module, core.Configurable):
    def __init__(self, 
                 gpu,
                 lm_type='bert',
                 gearnet_hidden_dim_size=512,
                 gearnet_hidden_dim_count=6,
                 gearnet_short_cut=True,
                 gearnet_concat_hidden=True,
                 lm_concat_to_output=False,
                 lm_short_cut=False,
                 lm_freeze_layer_count=None,
    ):
        super().__init__()
        Model, Tokenizer, pretrained_model_name, lm_layer_count = lm_type_map[lm_type]
        self.lm_layer_count = lm_layer_count
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = Tokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
        self.lm = Model.from_pretrained(pretrained_model_name).to(f'cuda:{gpu}')
        self.gearnet = models.GearNet(
            input_dim=self.lm.config.hidden_size,
            hidden_dims=([gearnet_hidden_dim_size] * gearnet_hidden_dim_count) if isinstance(gearnet_hidden_dim_size, int) else gearnet_hidden_dim_size,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=gearnet_concat_hidden,
            short_cut=gearnet_short_cut,
            readout="sum"
        ).to(f'cuda:{gpu}')
        self.input_dim = 21
        self.output_dim = self.gearnet.output_dim + self.lm.config.hidden_size if lm_concat_to_output else self.gearnet.output_dim
        self.lm_concat_to_output = lm_concat_to_output
        if lm_short_cut:
            assert self.gearnet.output_dim == self.lm.config.hidden_size, "lm_short_cut is only available when gearnet output dim is equal to lm hidden size"
        self.lm_short_cut = lm_short_cut
        self.gpu = gpu
        
        if lm_freeze_layer_count is not None:
            self.freeze_lm(freeze_layer_count=lm_freeze_layer_count)

    def forward(self, graph, _, all_loss=None, metric=None):
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]

        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.tokenizer(input, return_tensors='pt', padding=True).to(f'cuda:{self.gpu}')
        embedding_rpr = self.lm(**encoded_input)
        
        lm_residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            lm_residue_feature.append(emb[1:1+input_len[i]])
        
        lm_output = torch.cat(lm_residue_feature)

        gearnet_output = self.gearnet(graph, lm_output)

        final_output = torch.cat([gearnet_output['node_feature'], lm_output], dim=-1) if self.lm_concat_to_output else gearnet_output['node_feature']

        if self.lm_short_cut:
            final_output = final_output + lm_output
        return {
            "node_feature": final_output,
        }
        
    def get_parameters_with_discriminative_lr(self, lr=1e-5, lr_decay_factor=2):
        total_layers = self.lm_layer_count
        parameters = [
            {
                "params": item.parameters(), 
                "lr": lr / (lr_decay_factor ** (total_layers - i - 1))
            } 
            for i, item in enumerate(self.lm.encoder.layer)
        ] + [{"params": self.gearnet.parameters(), "lr": lr}]
        print('get_parameters_with_discriminative_lr:', parameters)
        return parameters
    
    def freeze_lm(self, freeze_layer_count=None):
        print('freeze_lm:', freeze_layer_count)
        # freeze the embeddings
        for param in self.lm.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in encoder
            for layer in self.lm.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
        
    
    def freeze_gearnet(self, freeze_all=False, freeze_layer_count=0):
        if freeze_all:
            for param in self.gearnet.parameters():
                param.requires_grad = False
        elif freeze_layer_count != 0:
            for layer in self.gearnet.layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.gearnet.edge_layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.gearnet.batch_norms[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False


class GearNetWrapModel(torch.nn.Module, core.Configurable):
    def freeze(self, freeze_all=False, freeze_layer_count=0):
        if freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_layer_count != 0:
            for layer in self.model.layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.model.edge_layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.model.batch_norms[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    def __init__(self, input_dim, hidden_dims, gpu):
        super().__init__()
        self.gpu = gpu
        self.model = models.GearNet(
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).cuda(gpu)
        self.output_dim = self.model.output_dim
        self.input_dim = input_dim

    def forward(self, graph, input, all_loss=None, metric=None):
        return self.model(graph, input, all_loss, metric)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)


class GVPWrapModel(torch.nn.Module, core.Configurable):
    '''
    Adapted from CPDModel of gvp-pytorch repository
    '''
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, gpu, num_layers=3, drop_rate=0.1, output_dim=1):
        super().__init__()
        logger.info(f'GVPWrapModel: node_in_dim: {node_in_dim}, node_h_dim: {node_h_dim}, edge_in_dim: {edge_in_dim}, edge_h_dim: {edge_h_dim}, num_layers: {num_layers}, drop_rate: {drop_rate}, output_dim: {output_dim}')
        self.gpu = gpu
        self.output_dim = output_dim

        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])
      
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, 
                             drop_rate=drop_rate, autoregressive=True) 
            for _ in range(num_layers))
        
        self.W_out = GVP(node_h_dim, (self.output_dim, 0), activations=(None, None))
    
    def forward(self, graph, gvp_data, all_loss=None, metric=None):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        h_V = (gvp_data['node_s'], gvp_data['node_v'])
        h_E = (gvp_data['edge_s'], gvp_data['edge_v'])
        edge_index = gvp_data['edge_index']
        seq = gvp_data['seq']

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V

        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E,
                        autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)

        return {
            "node_feature": logits
        }


class GVPEncoderWrapModel(torch.nn.Module, core.Configurable):
    '''
    Encoder-only architecture adapted from GVP model
    '''

    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, gpu, num_layers=3, drop_rate=0.1, output_dim=1):
        super().__init__()
        logger.info(
            f'GVPWrapModel: node_in_dim: {node_in_dim}, node_h_dim: {node_h_dim}, edge_in_dim: {edge_in_dim}, edge_h_dim: {edge_h_dim}, num_layers: {num_layers}, drop_rate: {drop_rate}, output_dim: {output_dim}')
        self.gpu = gpu
        self.output_dim = output_dim

        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        self.W_out = GVP(node_h_dim, (self.output_dim, 0),
                         activations=(None, None))

    def forward(self, graph, gvp_data, all_loss=None, metric=None):
        '''
        Forward pass for encoder-only architecture
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        '''
        h_V = (gvp_data['node_s'], gvp_data['node_v'])
        h_E = (gvp_data['edge_s'], gvp_data['edge_v'])
        edge_index = gvp_data['edge_index']

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        logits = self.W_out(h_V)

        return {
            "node_feature": logits
        }
