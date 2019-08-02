import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (GPT2DoubleHeadsModel, GPT2Tokenizer,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME,
                                     WarmupLinearSchedule)



logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
  
              
def load_dataset(tokenizer, dataset, num_prior = None):

    num_prior   = num_prior or 3
    df = pd.read_csv(dataset)

    # Clear empty and tokenize all input.
    text_data = [' '.join(t.split()) for t in df['Text'].tolist() if len(t.split()) > 1]

    output = []

    for i, text in enumerate(text_data):
        if i >= num_prior:
          pri = ' '.join(text_data[i-num_prior:i])
          nxt = text
          rdm = random.choice(text_data)
          if rdm == nxt:
            rdm = random.choice(text_data)
        
          s = random.choice([0,1])
          if s == 0:
            output.append((pri, nxt, rdm, s))
          else:
            output.append((pri, rdm, nxt, s))
            
    return output

      
      
def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
   
            try:
                with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
                with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
                input_ids[i, 0, :len(with_cont1)] = with_cont1
                input_ids[i, 1, :len(with_cont2)] = with_cont2
                mc_token_ids[i, 0] = len(with_cont1) - 1
                mc_token_ids[i, 1] = len(with_cont2) - 1
                lm_labels[i, 0, :len(with_cont1)] = with_cont1
                lm_labels[i, 1, :len(with_cont2)] = with_cont2
                mc_labels[i] = mc_label
            except Exception as e:
                print('Exception:', str(e))
                print('cont1:', str(with_cont1))
                print('cont2:', str(with_cont2))
                exit()

        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-medium',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.")
    parser.add_argument("--output_dir", default='fintuned_gpt', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--dataset', type=str, default='', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_prior', type=int, default=2)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset.
    # start_token, delimiter_token, clf_token

    special_tokens = ['<|endoftext|>', '<|endoftext|>', '<|cls|>']
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, unk_token = '<|endoftext|>', bos_token = '<|endoftext|>', eos_token = '<|endoftext|>', cls_token='<|cls|>')
    tokenizer.add_tokens(['<|cls|>']) 
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2DoubleHeadsModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(new_num_tokens=int(len(tokenizer)))
    
    model.to(device)


    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    logger.info("Encoding dataset...")

    train_dataset = load_dataset(tokenizer, args.dataset, num_prior = args.num_prior)
    eval_dataset = load_dataset(tokenizer, args.dataset, num_prior = args.num_prior)

    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
                        for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps //\
            (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader)\
            // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    for i, _ in enumerate(range(int(args.num_train_epochs))):
        print('Starting Epoch: {} of {}'.format(str(i+1), str(int(args.num_train_epochs))))
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            loss = args.lm_coef * losses[0] + losses[1]
            loss.backward()
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])

# Save a trained model

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = GPT2DoubleHeadsModel.from_pretrained(args.output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
    model.to(device)


if __name__ == '__main__':
    main()

