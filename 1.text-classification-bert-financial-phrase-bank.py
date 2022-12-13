import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,  AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

#Reading the data into a pandas dataframe
financial_data = pd.read_csv("financial_phrase_bank.csv", encoding='latin-1', 
names=['sentiment', 'NewsHeadline'])

#Function to encode the sentiment values
#The sentiments positive negative and neutral are mapped to 0,1,2
def encode_sentiments_values(df):
    
    possible_sentiments = df.sentiment.unique()
    sentiment_dict = {}
    
    for index, possible_sentiment in enumerate(possible_sentiments):
        sentiment_dict[possible_sentiment] = index
    
    # Encode all the sentiment values
    df['label'] = df.sentiment.replace(sentiment_dict)
    
    return df, sentiment_dict

# Encode the sentiment column
financial_data, sentiment_dict = encode_sentiments_values(financial_data)
financial_data.head()

# Create training and validation data
X_train, X_val, y_train, y_val = train_test_split(financial_data.index.values, 
                                                  financial_data.label.values, 
                                                  test_size = 0.15, 
                                                  random_state = 2022, 
                                                  stratify = financial_data.label.values)
                                                  
# Get the FinBERT Tokenizer
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert",do_lower_case=True)

# Encode the Training and Validation Data
#Hugging Face tokenizer
#batch_text_or_text_pairs (List[str], List[Tuple[str, str]], List[List[str]], List[Tuple[List[str], 
#List[str]]], and for not-fast tokenizers, also List[List[int]], List[Tuple[List[int], List[int]]]) — Batch of sequences or pair of sequences to be encoded. This can be a list of string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see details in encode_plus).
#add_special_tokens (bool, optional, defaults to True) — Whether or not to encode the sequences with the special tokens relative to their model.
#padding (bool, str or PaddingStrategy, optional, defaults to False) — Activates and controls padding. Accepts the following values:
#True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
#'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
#False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
#return_attention_mask (bool, optional) — Whether to return the attention mask. 
#If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the return_outputs attribute.
#return_tensors (str or TensorType, optional) — If set, will return tensors instead of list of python integers. Acceptable values are:
#'tf': Return TensorFlow tf.constant objects.
#'pt': Return PyTorch torch.Tensor objects.
#'np': Return Numpy np.ndarray objects.



encoded_data_train = finbert_tokenizer.batch_encode_plus(
    financial_data[financial_data.data_type=='train'].NewsHeadline.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 )

encoded_data_val = finbert_tokenizer.batch_encode_plus(
    financial_data[financial_data.data_type=='val'].NewsHeadline.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 )


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(financial_data[financial_data.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
sentiments_val = torch.tensor(financial_data[financial_data.data_type=='val'].label.values)

#Creating the train dataset 
#class:torch.utils.data.TensorDataset(*tensors)[source]
#Dataset wrapping tensors.
#Each sample will be retrieved by indexing tensors along the first dimension.
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

#Creating the test dataset 
dataset_val = TensorDataset(input_ids_val, attention_masks_val, sentiments_val)

#Defining the model
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",
                                                          num_labels=len(sentiment_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
                                                          
                                                          
batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)
                                   
#Setting the optimizer
optimizer1 = torch.optim.AdamW(model.parameters(),lr=5e-5,eps=1e-8)
optimizer2 = torch.optim.SGD(model.parameters(),lr=0.01)
optimizer3 = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.001)
optimizer4 = torch.optim.RMSprop(model.parameters(),lr=0.01, alpha=0.99, eps=1e-08, momentum=0.01)
optimizer5 = torch.optim.Adagrad(model.parameters(),lr=0.01, lr_decay=0, weight_decay=0)

epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer2, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)                                                                                             

                                                  
def evaluate(dataloader_val):
#torch.no_grad
#Context-manager that disabled gradient calculation.
     model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], [],
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)    
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer2.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'finetuned_finBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

#Reading Data from a csv into pandas dataframe    
financial_data = pd.read_csv("Financial_Phrase_Bank.csv", encoding='latin-1',names=['sentiment', 'NewsHeadline'])    
  
#Function to encode the sentiment values
#The sentiments positive negative and neutral are mapped to 0,1,2
def encode_sentiments_values(df):
    possible_sentiments = df.sentiment.unique()
    sentiment_dict = {}
    
    for index, possible_sentiment in enumerate(possible_sentiments):
        sentiment_dict[possible_sentiment] = index
    
    # Encode all the sentiment values
    df['label'] = df.sentiment.replace(sentiment_dict)
    
    return df, sentiment_dict  


# Create training and validation data
X_train, X_val, y_train, y_val = train_test_split(financial_data.index.values, 
                                                  financial_data.label.values, 
                                                  test_size = 0.15, 
                                                  random_state = 2022, 
                                                  stratify = financial_data.label.values)
                                                  
# Create the data type columns
financial_data.loc[X_train, 'data_type'] = 'train'
financial_data.loc[X_val, 'data_type'] = 'val'



# Get the FinBERT Tokenizer
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert",do_lower_case=True)

# Encode the Training and Validation Data
#Hugging Face tokenizer
#batch_text_or_text_pairs (List[str], List[Tuple[str, str]], List[List[str]], List[Tuple[List[str], 
#List[str]]], and for not-fast tokenizers, also List[List[int]], List[Tuple[List[int], List[int]]]) — Batch of sequences or pair of sequences to be encoded. This can be a list of string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see details in encode_plus).
#add_special_tokens (bool, optional, defaults to True) — Whether or not to encode the sequences with the special tokens relative to their model.
#padding (bool, str or PaddingStrategy, optional, defaults to False) — Activates and controls padding. Accepts the following values:
#True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
#'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
#False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
#return_attention_mask (bool, optional) — Whether to return the attention mask. 
#If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the return_outputs attribute.
#return_tensors (str or TensorType, optional) — If set, will return tensors instead of list of python integers. Acceptable values are:
#'tf': Return TensorFlow tf.constant objects.
#'pt': Return PyTorch torch.Tensor objects.
#'np': Return Numpy np.ndarray objects.



encoded_data_train = finbert_tokenizer.batch_encode_plus(
    financial_data[financial_data.data_type=='train'].NewsHeadline.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 )

encoded_data_val = finbert_tokenizer.batch_encode_plus(
    financial_data[financial_data.data_type=='val'].NewsHeadline.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 )


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(financial_data[financial_data.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
sentiments_val = torch.tensor(financial_data[financial_data.data_type=='val'].label.values)

#Creating the train dataset 
#class:torch.utils.data.TensorDataset(*tensors)[source]
#Dataset wrapping tensors.
#Each sample will be retrieved by indexing tensors along the first dimension.
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

#Creating the test dataset 
dataset_val = TensorDataset(input_ids_val, attention_masks_val, sentiments_val)                                                  
                                                      
batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], [],
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)    
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer2.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'finetuned_finBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    #print(train_acc = torch.sum(y_pred == true_vals))


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in sentiment_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    acc = []
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        acc.append((len(y_preds[y_preds==label])) / (len(y_true)))
    print("Model Accuracy: ", np.mean(acc)*100)

def accuracy(preds,labels):  
    label_dict_inverse = {v: k for k, v in sentiment_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    acc = []
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(y_preds)
        print(y_true)
        #print(f'Accuracy: {(y_preds[y_preds==label])}/{len(y_preds)}\n')

# Load the best model & Make Predictions

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",
                                                          num_labels=len(sentiment_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('finetuned_finBERT_epoch_1.model', 
                                 map_location=torch.device('cpu')))

_, predictions, true_vals = evaluate(dataloader_validation)

accuracy_per_class(predictions, true_vals)
accuracy(predictions, true_vals)            
