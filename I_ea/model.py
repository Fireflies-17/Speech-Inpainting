from transformers import HubertModel, HubertConfig
import torch.nn as nn


class CustomHubert(nn.Module):
    def __init__(self, device):
        super(CustomHubert, self).__init__()
        self.base_model = HubertModel.from_pretrained(
            "facebook/hubert-large-ls960-ft")
        # Freeze the Hubert layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.samplerate = 16000
        self.device = device

    def forward(self, x):
        x = self.base_model(x[0]).last_hidden_state
        return x


class CustomModel(nn.Module):
    def __init__(self, codebook_dim, type='large', load_pretrained=True, train_encoder=False, loss_function=''):
        super().__init__()
        import pdb
        # pdb.set_trace()
        if type == 'base':
            print("Initializing model from HuBERT base\n")
            pretrained_model_name = "facebook/hubert-base-ls960"
        else:
            print("Initializing model from HuBERT large\n")
            pretrained_model_name = "facebook/hubert-large-ls960-ft"
        pretrained_model = HubertModel.from_pretrained(pretrained_model_name)
        if load_pretrained:
            print("Initializing transformer encoder + CNN prenet from HugginFace model")
            self.base_model = pretrained_model
        else:
            print(
                "Initializing only the CNN prenet from HugginFace model (and not the transformer encoder)")
            model_config = HubertConfig.from_pretrained(pretrained_model_name)
            self.base_model = HubertModel(config=model_config)
            state_dict = pretrained_model.state_dict()
            new_state_dict = self.base_model.state_dict()

            for key in state_dict.keys():
                if not key.startswith('encoder'):
                    new_state_dict[key] = state_dict[key]
            self.base_model.load_state_dict(new_state_dict)

        self.last_hidden_dim = self.base_model.config.hidden_size

        for param in self.base_model.parameters():
            param.requires_grad = False
        if train_encoder:
            for param in self.base_model.encoder.parameters():
                param.requires_grad = True

        # update some configs in Hubert, like masking:
        self.base_model.config.mask_time_prob = 0
        self.base_model.config.mask_feature_prob = 0
        self.base_model.config.mask_feature_length = 0
        self.base_model.config.mask_feature_min_masks = 0
        self.base_model.config.mask_time_length = 0
        self.base_model.config.mask_time_min_masks = 0

        # check the updated config
        config = self.base_model.config

        # self.layer_norm = nn.LayerNorm(self.base_model.config.hidden_size)
        if loss_function == 'softmax':
            self.final_layers = nn.Sequential(
                nn.LayerNorm(self.last_hidden_dim),
                nn.Linear(self.last_hidden_dim, 100)
            )
        else:
            self.final_layers = nn.Sequential(
                nn.LayerNorm(self.last_hidden_dim),
                nn.Linear(self.last_hidden_dim, codebook_dim)
            )

    def forward(self, input_values, attention_mask=None):
        if attention_mask is not None:
            outputs = self.base_model(
                input_values, attention_mask=attention_mask)
        else:
            outputs = self.base_model(input_values)
        # shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.final_layers(sequence_output)
        return sequence_output
