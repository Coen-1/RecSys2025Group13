import torch
from collections import OrderedDict
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

PRETRAINED_CKPT_PATH = 'pretrain_ckpt/pytorch_model.bin'
LONGFORMER_CKPT_PATH = 'longformer_ckpt/longformer-base-4096.bin'
LONGFORMER_TYPE = 'allenai/longformer-base-4096'
RECFORMER_OUTPUT_PATH = 'pretrain_ckpt/recformer_pretrain_ckpt.bin'
RECFORMERSEQREC_OUTPUT_PATH = 'pretrain_ckpt/seqrec_pretrain_ckpt.bin'

input_file = PRETRAINED_CKPT_PATH
lightning_ckpt = torch.load(input_file)
state_dict = lightning_ckpt['state_dict']

# Clean the keys from the lightning checkpoint
cleaned_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[len('model.'):]
        cleaned_state_dict[new_key] = value
    else:
        cleaned_state_dict[key] = value
state_dict = cleaned_state_dict

longformer_file = LONGFORMER_CKPT_PATH
longformer_state_dict = torch.load(longformer_file)

# The lightning checkpoint already contains the longformer weights,
# but the original script adds the word embeddings separately.
# We will replicate this logic to be safe.
state_dict['longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

output_file = RECFORMER_OUTPUT_PATH
new_state_dict = OrderedDict()

for key, value in state_dict.items():

    if key.startswith('longformer.'):
        new_key = key[len('longformer.'):]
        new_state_dict[new_key] = value

config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
model = RecformerModel(config)
model.load_state_dict(new_state_dict)

print('Convert successfully.')
torch.save(new_state_dict, output_file)



output_file = RECFORMERSEQREC_OUTPUT_PATH
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    new_state_dict[key] = value

config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
model = RecformerForSeqRec(config)

model.load_state_dict(new_state_dict, strict=False)

print('Convert successfully.')
torch.save(new_state_dict, output_file)
