import tokenization

import numpy as np


def get_start_end_pos(class_type, token_label_ids, max_seq_length):
  if class_type == 'copy_value' and 1 not in token_label_ids:
    raise ValueError('Copy value but token_label not detected.')
  if class_type != 'copy_value':
    start_pos = 0
    end_pos = 0
  else:
    start_pos = token_label_ids.index(1)
    end_pos = max_seq_length - 1 - token_label_ids[::-1].index(1)
    for i in range(max_seq_length):
      if i >= start_pos and i <= end_pos:
        assert token_label_ids[i] == 1
      else:
        assert token_label_ids[i] == 0
  return start_pos, end_pos


def get_token_label_ids(token_labels_a, token_labels_b, max_seq_length):
  token_label_ids = []
  #token_label_ids.append(0)

  for token_label in token_labels_a:
    token_label_ids.append(token_label)

  #token_label_ids.append(0)

  for token_label in token_labels_b:
    token_label_ids.append(token_label)

  #token_label_ids.append(0)

  while len(token_label_ids) < max_seq_length:
    token_label_ids.append(0)

  assert len(token_label_ids) == max_seq_length
  return token_label_ids


def tokenize_text_and_label(text, text_label_dict, slot, tokenizer, 
        slot_value_dropout=0.3):
  joint_text_label = [0 for _ in text_label_dict[slot]] # joint all slots' label
  for slot_text_label in text_label_dict.values():
    for idx, label in enumerate(slot_text_label):
      if label == 1:
        joint_text_label[idx] = 1

  text_label = text_label_dict[slot]
  tokens = []
  token_labels = []
  for token, token_label, joint_label in zip(text, text_label, joint_text_label):
    token = tokenization.convert_to_unicode(token)
    sub_tokens = tokenizer.tokenize(token)
    if slot_value_dropout == 0.0 or joint_label == 0:
      tokens.extend(sub_tokens)
    else:
      rn_list = np.random.random_sample((len(sub_tokens),))
      for rn, sub_token in zip(rn_list, sub_tokens):
        if rn > slot_value_dropout:
          tokens.append(sub_token)
        else:
          tokens.append('[UNK]')

    token_labels.extend([token_label for _ in sub_tokens])
  assert len(tokens) == len(token_labels)
  return tokens, token_labels

