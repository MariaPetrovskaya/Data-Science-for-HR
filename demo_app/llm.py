import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertConfig


class TransformerRegrModel(nn.Module):
    def __init__(self, base_transformer_model: str, num_classes: int):
        super().__init__()
        self.tr_model = base_transformer_model
        self.num = num_classes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.tr_model not in ['rubert', 'base']:
            raise Exception('unknown model')
        elif self.tr_model == 'rubert':
            self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            self.config = BertConfig.from_pretrained("cointegrated/rubert-tiny2", output_hidden_states=True,
                                                     output_attentions=True)
        elif self.tr_model == 'base':
            self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base", model_max_length=512)
            self.config = BertConfig.from_pretrained("ai-forever/ruBert-base", output_hidden_states=True,
                                                     output_attentions=True)
        self.model = AutoModel.from_config(self.config)
        self.a1 = nn.ReLU()
        self.classifier_1 = nn.Linear(self.model.pooler.dense.out_features, self.num)
        # self.classifier_dropout = nn.Dropout(p=0.2)
        # self.classifier_2 = nn.Linear(128, self.num)

    def forward(self, inputs):
        t = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        tokens = self.tokenizer.convert_ids_to_tokens(t['input_ids'][0])
        model_output = self.model(**{k: v.to(self.device) for k, v in t.items()})
        attentions = torch.cat(model_output['attentions']).to('cpu')
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        outputs = self.a1(embeddings)
        outputs = self.classifier_1(outputs)
        # outputs = self.classifier_dropout(outputs)
        # outputs = self.a1(outputs)
        # outputs = self.classifier_dropout(outputs)
        # outputs = self.classifier_2(outputs)

        return outputs, tokens, attentions
