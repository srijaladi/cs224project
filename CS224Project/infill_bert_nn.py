import torch
import tensorflow as tf
import openai
import re
import numpy as np
from matplotlib import pyplot as plt
import requests
import os
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TFTrainer, TFTrainingArguments
from transformers import TFAutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import AutoTokenizer
import evaluate
# from nltk.corpus import stopwords
from datasets import Dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader

TRAIN_VALID_SPLIT = 0.7
BASE_MODEL_NAME = 'bert-base-uncased'
FT_MODEL_NAME = "distilbert-base-uncased"
MODEL_ENGINE = "text-davinci-003"
RANK_CUTTOFF = 25
NUM_MASK = 3
EARLY_STOP = 100

BASE_MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
FT_MODEL = AutoModelForMaskedLM.from_pretrained(FT_MODEL_NAME)
BASE_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
FT_TOKENIZER = AutoTokenizer.from_pretrained(FT_MODEL_NAME)

recorded_losses = []

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

def embedding_coherence_score_single (embedding):
  return 1
  new_sentence = FT_TOKENIZER.decode(embedding)
  return sentence_coherence_score_single(new_sentence)

def model_score_calc (inputs, token_logits):
  batch_size = len (token_logits)
  average_coherence = 0
  average_perplexity = 0
  for i in range (batch_size):
    mask_token_index = np.argwhere(inputs["input_ids"] == FT_TOKENIZER.mask_token_id)[0, 1]
    mask_token_logits = token_logits[i, mask_token_index, :]
    top_token = torch.argsort(-mask_token_logits)[0]
    embedding_copy = torch.clone(inputs['input_ids'][i])    
    embedding_copy[mask_token_index] = top_token
    average_coherence += embedding_coherence_score_single (embedding_copy)
    average_perplexity += compute_perplexity(embedding_copy)[0].item()
  average_coherence /= batch_size
  average_perplexity /= batch_size
  return average_coherence, average_perplexity

def get_prob(model, encoded_sentence, tf= False):    
    isGPT = BASE_MODEL_NAME[:4] == 'gpt2'
    def get_word_prob(ids_so_far, true_token):
        with torch.inference_mode():
            if len(ids_so_far.size()) == 1 and not(isGPT):
                ids_so_far = torch.unsqueeze(ids_so_far,0)
            end_model = model(input_ids = ids_so_far)
            logits = end_model.logits
            all_probs = torch.nn.functional.softmax(logits, dim = -1)
            if isGPT: probability = all_probs[-1][true_token]
            else: probability =  all_probs[0][-1][true_token]  
            return probability
        
    def tf_get_word_prob(ids_so_far, true_token):
      if len(ids_so_far) == 1 and not(isGPT):
          ids_so_far = np.expand_dims(ids_so_far,0)
      end_model = FT_MODEL(input_ids = ids_so_far)
      logits = end_model.logits
      all_probs = tf.nn.softmax(logits, axis = -1)

      if isGPT: probability = all_probs[-1][true_token]
      else: probability =  tf.reshape(all_probs, [-1])[true_token.item()]
          
      return probability
    
    all_probs = np.zeros(len(encoded_sentence))
    total_log_prob = 0
    for i in range(0,len(encoded_sentence)):
        word_cond_prob = get_word_prob(encoded_sentence[:i+1], encoded_sentence[i]) if not tf else tf_get_word_prob(encoded_sentence[:i+1], encoded_sentence[i])
        all_probs[i] = word_cond_prob.numpy()
        total_log_prob += np.log(word_cond_prob)
    
    return total_log_prob, all_probs

# REMOVE STOP WORDS
def compute_perplexity(encoded_sentence, tf = False):
    base_log_prob, base_each_prob = get_prob(BASE_MODEL, encoded_sentence, tf = tf)
    # print(base_log_prob)
    N = len(encoded_sentence)
    
    overall_perplexity = 2 ** (-(1/N) * base_log_prob/np.log(2)) #(1/base_prob) ** (1/len(encoded_sentence))
    # print(overall_perplexity)
    return overall_perplexity, base_each_prob
  
def mask_ith_word (sentence, i):
  words = sentence.split(" ")
  words[i] = "[MASK]"
  return " ".join(words)

def get_mask_indicies (encoded_sentence, tf = False):
  sentence_perplexity, prob_each_index = compute_perplexity(encoded_sentence, tf = tf)
  indexes_by_prob = [[p,i] for i,p in enumerate(prob_each_index)]
  indexes_by_prob = indexes_by_prob [1: len(indexes_by_prob) - 1]
  indexes_by_prob = sorted(indexes_by_prob)

  return [tu[1] for tu in indexes_by_prob[:NUM_MASK]]

def similarity_score_single(embed1, embed2):
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    numerator = np.dot(embed1, embed2)
    denominator = norm1 * norm2
    return numerator/denominator

def get_most_similar_token (tokens, original_encoding, index):
   max_sim_score = 0
   top_token = None
   for token in tokens:
      new_encoding = [i for i in original_encoding]
      new_encoding [index] = token
      sim_score = similarity_score_single(new_encoding, original_encoding)
      if sim_score > max_sim_score:
        sim_score = max_sim_score
        top_token = token
   return top_token

class FinetunedBartTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs):
      token_logits = model(**inputs).logits
      cross_entropy_loss = model(**inputs).loss
      coherence, perplexity = model_score_calc (inputs, token_logits)
      bias = 10 * (1.5 - coherence) * np.exp(-perplexity/200)
      recorded_losses.append(bias * cross_entropy_loss.item())
      return bias * cross_entropy_loss
class TFBERTDataset(Dataset):
  def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels
  def __len__(self):
      return len(self.labels)
  def __getitem__(self, idx):
      item = {key: tf.Tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = tf.Tensor(self.labels[idx])
      return item

class BERTDataset(Dataset):
  def __init__(self, encodings):
      self.encodings = encodings
  def __len__(self):
      return len(self.encodings.input_ids)
  def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      return item
    
def prep_data_set (training_data):
  dataset_train =  FT_TOKENIZER (training_data, return_tensors="pt", padding=True)
  dataset_train["labels"] = np.copy(dataset_train["input_ids"])

  # mask lowest perplexity token
  for i, encoding in enumerate(dataset_train["input_ids"]):
    base_encoding = BASE_TOKENIZER (training_data[i], return_tensors = 'pt')['input_ids']
    mask_idx = get_mask_indicies (base_encoding[0])[0]
    encoding[mask_idx] = FT_TOKENIZER.mask_token_id
  return BERTDataset(dataset_train)

def fine_tune (model, training_data):
  inputs = prep_data_set (training_data)
  print (inputs)
   
  training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs = 3,
        gradient_accumulation_steps = 1,
        per_device_train_batch_size = 8,
        learning_rate = 5e-5,
        logging_steps = 400
    )
  
  trainer = FinetunedBartTrainer(
    model=model,
    args=training_args,
    train_dataset=inputs
   )
  trainer.train()
  trainer.save_model("./fine-tuned-model")
  

if __name__ == "__main__":
  print ("\n\n\n BEGINGING RUN\n")
  training_data, validation_data  = get_preprocessed_data ()

  # BEGIN FINE TUNING
  fine_tune (FT_MODEL, training_data)

  y = -np.array(recorded_losses)
  x = range (1, len(recorded_losses) + 1)
  plt.scatter(x, y)

  plt.title("Loss values for BERT Fine-Tuning over batches")
  plt.xlabel("Training Batch Number")
  plt.ylabel("Custom loss Value")

  plt.savefig('fine_tuning_loss_graph.png')

  old_new_dict = {}
  print ("BEGINGING EVAL\n")
  for index, sentence in enumerate(validation_data):
    encoding = BASE_TOKENIZER (sentence, return_tensors = 'pt')['input_ids']
    mask_indicies = get_mask_indicies (encoding[0])
    inputs = FT_TOKENIZER (sentence, return_tensors="pt")
    new_sentence = "" + sentence
    for i in range (NUM_MASK):
      if (i >= len(mask_indicies) or mask_indicies[i] >= len(inputs['input_ids'][0])):
        continue
      inputs['input_ids'][0][mask_indicies[i]] = FT_TOKENIZER.mask_token_id
      token_logits = FT_MODEL(**inputs).logits
      mask_token_index = mask_indicies[i]
      mask_token_logits = token_logits[0, mask_token_index, :]
      top_tokens = torch.argsort(-mask_token_logits)[:RANK_CUTTOFF].tolist()
      top_token = get_most_similar_token (top_tokens, inputs['input_ids'][0], mask_indicies[i])
      inputs['input_ids'][0][mask_indicies[i]] = top_token
      new_sentence = FT_TOKENIZER.decode(inputs['input_ids'][0])
    with open('fine_tuned_sentences.txt', 'a') as f:
      print(f"original:{sentence}\nmodified: {new_sentence}\n", file=f)

    old_new_dict [sentence] = new_sentence
    # if index >= EARLY_STOP:
    #   break
  print ("ENDING EVAL\n")
  print (old_new_dict)