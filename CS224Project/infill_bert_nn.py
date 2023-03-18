import torch
import tensorflow as tf
import openai
import re
import numpy as np
from matplotlib import pyplot as plt
import requests
import os
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TFAutoModelForMaskedLM
from transformers import AutoTokenizer
import evaluate
# from nltk.corpus import stopwords
from datasets import Dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader

TRAIN_VALID_SPLIT = 0.7
BASE_MODEL_NAME = "distilbert-base-uncased"
RANK_CUTTOFF = 25
NUM_MASK = 3

BASE_MODEL = TFAutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

class FinetunedBartTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs):
       return 0

def get_preprocessed_data ():
  all_train_sents = []
  training_data = []
  validation_data = []
  with open('train_sents.txt') as f:
      lines = f.readlines()
      all_train_sents = [line.strip('\n') for line in lines]
  np.random.seed(123)
  np.random.shuffle(all_train_sents)
  split_idx = int(TRAIN_VALID_SPLIT * len(all_train_sents))
  training_data = all_train_sents[:split_idx]
  validation_data = all_train_sents[split_idx:]
  return training_data, validation_data


def sentence_embedding(input_sentence, return_type = "torch"):
  response = openai.Embedding.create(
    input=input_sentence,
    engine="text-similarity-davinci-001")
  res = response.data[0]['embedding']
  
  if return_type.lower() == "np" or return_type.lower() == "numpy":
      return np.array(res)
  elif return_type.lower() == "list":
      return res
  else:
      return torch.tensor(res)

def similarity_score_single(sentence1, sentence2):
    embed1 = sentence_embedding(sentence1, "torch")
    embed2 = sentence_embedding(sentence2, "torch")
    norm1 = torch.sqrt(torch.sum(embed1 * embed1))
    norm2 = torch.sqrt(torch.sum(embed2 * embed2))
    numerator = torch.dot(embed1, embed2)
    denominator = norm1 * norm2
    return numerator/denominator

def sentence_coherence_score_single(input_sentence):
    return 0.5
    modified_prompt = "Evaluate the coherence score of this sentence as a value between 0 and 1:\n\n" + input_sentence
    response = openai.Completion.create(
      model=MODEL_ENGINE,
      prompt=modified_prompt,
      temperature=0,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    res = response.choices[0]['text'].strip()
    return float(res)

def get_prob(model, encoded_sentence):    
    isGPT = BASE_MODEL_NAME[:4] == 'gpt2'
    
    def get_word_prob(ids_so_far, true_token):
        with torch.inference_mode():
            if len(ids_so_far.size()) == 1 and not(isGPT):
                ids_so_far = np.array(torch.unsqueeze(ids_so_far,0))
            end_model = model(input_ids = ids_so_far)
            logits = end_model.logits
            all_probs = tf.nn.softmax(logits, axis = -1)

            if isGPT: probability = all_probs[-1][true_token]
            else: probability =  tf.reshape(all_probs, [-1])[true_token.item()]
                
            return probability
    
    all_probs = np.zeros(len(encoded_sentence))
    total_log_prob = 0
    for i in range(0,len(encoded_sentence)):
        word_cond_prob = get_word_prob(encoded_sentence[:i+1], encoded_sentence[i])
        all_probs[i] = word_cond_prob.numpy()
        total_log_prob += np.log(word_cond_prob)
    
    return total_log_prob, all_probs

def compute_perplexity(full_sentence, encoded_sentence):
    base_log_prob, base_each_prob = get_prob(BASE_MODEL, encoded_sentence)
    # print(base_log_prob)
    N = len(encoded_sentence)
    
    overall_perplexity = 2 ** (-(1/N) * base_log_prob/np.log(2)) #(1/base_prob) ** (1/len(encoded_sentence))
    # print(overall_perplexity)
    return overall_perplexity, base_each_prob
  
def mask_ith_word (sentence, i):
  words = sentence.split(" ")
  words[i] = "[MASK]"
  return " ".join(words)

def get_mask_indicies (full_sentence, encoded_sentence):
  sentence_perplexity, prob_each_index = compute_perplexity(full_sentence, encoded_sentence)
  indexes_by_prob = [[p,i] for i,p in enumerate(prob_each_index)]
  indexes_by_prob = sorted(indexes_by_prob)

  print (indexes_by_prob)

  return [tu[1] for tu in indexes_by_prob[:NUM_MASK]]

if __name__ == "__main__":
  print ("\n\n\n BEGINGING RUN\n")
  training_data, validation_data  = get_preprocessed_data ()

  sentence = validation_data[0]
  print (sentence)
  encoding = TOKENIZER (sentence, return_tensors = 'np')['input_ids'][0]
  mask_indicies = get_mask_indicies (sentence, encoding)
  new_sentence = "" + sentence
  for i in range (NUM_MASK):
    new_encoding = TOKENIZER (new_sentence, return_tensors = 'np')['input_ids']
    masked_sentence = mask_ith_word (new_sentence, mask_indicies[i])
    print (new_encoding)
    inputs = TOKENIZER (masked_sentence, return_tensors="np")
    token_logits = BASE_MODEL(**inputs).logits
    mask_token_index = np.argwhere(inputs["input_ids"] == TOKENIZER .mask_token_id)[0, 1]
    print ("MASKED TOKEN", mask_token_index, mask_indicies[i])
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_token = np.argsort(-mask_token_logits)[0]
    new_sentence = masked_sentence.replace(TOKENIZER .mask_token, TOKENIZER .decode([top_token]))
    print (new_sentence)



  # text = mask_ith_word(training_data[0], 0)

  # inputs = TOKENIZER (text, return_tensors="np")
  # token_logits = BASE_MODEL(**inputs).logits
  # mask_token_index = np.argwhere(inputs["input_ids"] == TOKENIZER .mask_token_id)[0, 1]
  # mask_token_logits = token_logits[0, mask_token_index, :]

  # top_tokens = np.argsort(-mask_token_logits)[:RANK_CUTTOFF].tolist()

  # for token in top_tokens:
  #     print(f">>> {text.replace(TOKENIZER .mask_token, TOKENIZER .decode([token]))}")