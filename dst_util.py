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

  if len(token_label_ids) != max_seq_length:
      print(token_label_ids)
      print(max_seq_length)
      print(len(token_label_ids))
      token_label_ids = token_label_ids[:max_seq_length]
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

def get_bert_input(tokens_a, tokens_b, max_seq_length, tokenizer):
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []

  tokens.append("[CLS]")
  segment_ids.append(0)

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)

  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  input_ids = input_ids[:max_seq_length]
  input_mask = input_mask[:max_seq_length]
  segment_ids = segment_ids[:max_seq_length]

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return tokens, input_ids, input_mask, segment_ids



