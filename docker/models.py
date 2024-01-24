
import torch
from torchdrug import core, models
from transformers import AutoTokenizer, EsmModel


def separate_alphabets(text):
    separated_text = ""
    for char in text:
        if char.isalpha():
            separated_text += char + " "
    return separated_text.strip()
class LMGearNetModel(torch.nn.Module, core.Configurable):
    def __init__(self, gearnet_hidden_dim_size=512, gearnet_hidden_dim_count=4):
        super().__init__()
        model_name = 'facebook/esm2_t30_150M_UR50D'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.lm = EsmModel.from_pretrained(model_name)
        self.gearnet = models.GearNet(
            input_dim=self.lm.config.hidden_size,
            hidden_dims=[gearnet_hidden_dim_size] * gearnet_hidden_dim_count,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum"
        )
        self.input_dim = 21
        self.output_dim = self.gearnet.output_dim

    def forward(self, graph, _, all_loss=None, metric=None):
        # print("sequence: ", graph.to_sequence())
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]

        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.tokenizer(input, return_tensors='pt', padding=True)
        embedding_rpr = self.lm(**encoded_input)

        lm_residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            lm_residue_feature.append(emb[1:1+input_len[i]])

        lm_output = torch.cat(lm_residue_feature)

        gearnet_output = self.gearnet(graph, lm_output)

        final_output = gearnet_output['node_feature']

        return {
            "node_feature": final_output,
        }

    def freeze_lm(self, freeze_all=True, freeze_layer_count=None):
        if freeze_all:
            # freeze the entire bert model
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            # freeze the embeddings
            for param in self.lm.embeddings.parameters():
                param.requires_grad = False
            if freeze_layer_count != -1:
                # freeze layers in bert_model.encoder
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
