# freeze_bert in https://github.com/aws-samples/lm-gvp/blob/0b7a6d96486e2ee222929917570432296554cfe7/lmgvp/modules.py#L47

from transformers import BertModel, BertTokenizer, AutoTokenizer, EsmModel
from torchdrug import core
import torch
def separate_alphabets(text):
    separated_text = ""
    for char in text:
        if char.isalpha():
            separated_text += char + " "
    return separated_text.strip()

def _freeze_bert(
    bert_model: BertModel, freeze_bert=True, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)

    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether or not to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None


# Cusom model Wrapping BERT: check https://torchdrug.ai/docs/notes/model.html
class BertWrapModel(torch.nn.Module, core.Configurable):
    def __init__(self, gpu, freeze_bert, freeze_layer_count):
        super().__init__()
        self.gpu = gpu
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False)
        self.bert_model = BertModel.from_pretrained(
            "Rostlab/prot_bert").to(f'cuda:{gpu}')
        _freeze_bert(self.bert_model,
                     freeze_bert=freeze_bert,
                     freeze_layer_count=freeze_layer_count)
        self.input_dim = 21
        self.output_dim = self.bert_model.config.hidden_size

    def forward(self, graph, _, all_loss=None, metric=None):
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]
        # print(f'bert graph: {graph}')
        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.bert_tokenizer(
            input, return_tensors='pt', padding=True).to(f'cuda:{self.gpu}')
        embedding_rpr = self.bert_model(**encoded_input)

        residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            residue_feature.append(emb[1:1+input_len[i]])

        # print(f'bert residue_feature shape: {residue_feature[0].shape}, len: {len(residue_feature)}')
        x = torch.cat(residue_feature)
        # print(f'bert x shape: {x.shape}')
        return {"residue_feature": x}


class EsmWrapModel(torch.nn.Module, core.Configurable):
    def __init__(self, model_type, gpu, freeze_esm, freeze_layer_count):
        super().__init__()
        if model_type == 'esm-t33':
            name = "facebook/esm2_t33_650M_UR50D"
        elif model_type == 'esm-t36':
            name = "facebook/esm2_t36_3B_UR50D"
        elif model_type == 'esm-t48':
            name = "facebook/esm2_t48_15B_UR50D"
        elif model_type == 'esm-t30':
            name = "facebook/esm2_t30_150M_UR50D"
        elif model_type == 'esm-t12':
            name = "facebook/esm1_t12_35M_UR50D"
        elif model_type == 'esm-t6':
            name = "facebook/esm1_t6_8M_UR50D"      
            
        self.esm_tokenizer = AutoTokenizer.from_pretrained(name)
        self.gpu = gpu
        self.esm_model = EsmModel.from_pretrained(name).to(f'cuda:{gpu}')
        _freeze_bert(self.esm_model,
                     freeze_bert=freeze_esm,
                     freeze_layer_count=freeze_layer_count)
        self.input_dim = 21
        self.output_dim = self.esm_model.config.hidden_size

    def forward(self, graph, _, all_loss=None, metric=None):
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]
        # print(f'esm graph: {graph}')
        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.esm_tokenizer(
            input, return_tensors='pt', padding=True).to(f'cuda:{self.gpu}')
        embedding_rpr = self.esm_model(**encoded_input)

        residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            residue_feature.append(emb[1:1+input_len[i]])
        # print(f'esm residue_feature shape: {residue_feature[0].shape}, len: {len(residue_feature)}')
        x = torch.cat(residue_feature)
        return {"residue_feature": x}
