from transformers import get_constant_schedule, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup , AutoModelForSeq2SeqLM , AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
import evaluate
import json
import os
import shutil
import time
import datetime
from math import floor
import numpy as np

def create_dataset(datasets: list, datasetNames: list, config):
    def downsample(ds, sf):
        shardList = []
        tenths = floor((sf * 10))
        for i in range(tenths):
            sha = ds.shard(num_shards=10, index=i)
            if len(sha) > 0:
                shardList.append(sha)
        extraShard = ds.shard(num_shards=10, index=9)
        hundredths = floor(100 * (sf - tenths / 10))
        for i in range(hundredths):
            sha = extraShard.shard(num_shards=10, index=i)
            if len(sha) > 0:
                shardList.append(sha)
        return concatenate_datasets(shardList)

    def sample(ds, sf):
        if sf == 0:
            ds = ds.filter(lambda ex: False)
        elif sf < 1:
            ds = downsample(ds.shuffle(), sf)
        elif sf > 1:
            ds = [ds] * floor(sf)
            if sf % 1 != 0:
                ds.append(downsample(ds[0].shuffle(), sf - floor(sf)))
            ds = concatenate_datasets(ds)
        return ds

    dl = {split: datasets[0][split].filter(lambda ex: False) for split in datasets[0]} #  unpacks the dataset into a test and train set 
    for dataset, name in zip(datasets, datasetNames): #  no need for for loop
        # dataset 
        # datsset name 
        for split in dataset:
            datasetList = []
            ds = dataset[split]
            if split == "train":
                dsSampled = sample(ds, 1)
            elif split == "test":
                dsSampled = sample(ds, 1)
            datasetList.append(dsSampled)
            dt = concatenate_datasets(datasetList)
            dt = dt.rename_column("English", "labels")   # outpuy 
            dt = dt.rename_column("Luganda", "input_ids")  #input
            dl[split] = concatenate_datasets([dl[split], dt])
    return dl

def save_results_to_json(train_loss, eval_loss, bleu, config, dir):
    results = {"train_loss": train_loss, "eval_loss": eval_loss, "bleu": bleu, "config": config}
    json_object = json.dumps(results)
    with open(dir, "w") as outfile:
        outfile.write(json_object)

def train(model, tokenizer, dataset, config):
    max_input_length = 128
    max_target_length = 128

    def preprocess(examples):
        model_inputs = tokenizer(text=examples["input_ids"], text_target=examples["labels"], max_length=max_input_length, truncation=True, padding=False)
        return model_inputs


    model_input = dataset["train"].map(preprocess, batched=True)
    eval_input = dataset["test"].map(preprocess, batched=True)

    sacrebleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    num_train_epochs = config["training_epochs"]
    learning_rate = 5e-5
    per_device_train_batch_size = config["batch_size"]
    per_device_eval_batch_size = config["batch_size"]
    gradient_accumulation = 10
    epoch_per_save = 5
    epoch_per_eval = 3
    loss_log_steps = 100
    max_num_checkpoints = 5

    def custom_collate(batch):
        model_inputs = {"input_ids": [torch.tensor(d["input_ids"]) for d in batch], "labels": [torch.tensor(d["labels"]) for d in batch], "attention_mask": [torch.tensor(d["attention_mask"]) for d in batch]}
        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
        model_inputs["labels"] = pad_sequence(model_inputs["labels"], batch_first=True, padding_value=-100)
        model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], batch_first=True, padding_value=0)
        return model_inputs

    train_dataloader = DataLoader(model_input, shuffle=True, batch_size=per_device_train_batch_size, collate_fn=custom_collate)
    eval_dataloader = DataLoader(eval_input, shuffle=True, batch_size=per_device_eval_batch_size, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if config["LRschedule"] == "constant":
        lrSchedule = get_constant_schedule(optimizer=optimizer)
    elif config["LRschedule"] == "cosine":
        lrSchedule = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_epochs * len(train_dataloader))
    elif config["LRschedule"] == "linear":
        lrSchedule = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_epochs * len(train_dataloader))
    else:
        lrSchedule = get_constant_schedule(optimizer=optimizer)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation)
    model, optimizer, lrSchedule, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, lrSchedule, train_dataloader, eval_dataloader)

    outputDir = config["output_dir"] 
    checkpointDir = outputDir + "/Checkpoints"
    os.mkdir(outputDir)
    os.mkdir(checkpointDir)

    progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

    train_loss_log = []
    eval_data = []
    bleu_data = []
    step = 0
    epoch = -1

    def lossEval(model, eval_dataloader, eval_data, config):
        model.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            labels = batch.pop("labels")
            with torch.no_grad():
                loss, _ = model(**batch, labels=labels, use_cache=False)[:2]
            loss = accelerator.gather(loss)
            eval_loss += loss.item()
            if time.time() > config["maxTime"]:
                break
        eval_loss /= len(eval_dataloader)
        eval_data.append(((step, epoch + 1), eval_loss))
        return eval_data

    def bleuEval(model, eval_dataloader, bleu_data, config):
        model.eval()
        sacrebleu = evaluate.load("sacrebleu", experiment_id=config["PBS_ID"])
        for batch in eval_dataloader:
            labels = batch.pop("labels")
            with torch.no_grad():
                preds = accelerator.unwrap_model(model).generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=max_input_length)
                preds = accelerator.gather(preds).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                labels, preds = postprocess_text(preds, labels)
                sacrebleu.add_batch(predictions=preds, references=labels)
            if time.time() > config["maxTime"]:
                break
        metrics = sacrebleu.compute()
        bleu_data.append(((step, epoch + 1), metrics["score"]))
        return bleu_data

    eval_data = lossEval(model, eval_dataloader, eval_data, config)

    for epoch in range(num_train_epochs):
        model.train()
        total_train_loss = 0
        batch_idx = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                labels = batch.pop("labels")
                loss, _ = model(**batch, labels=labels, use_cache=False)[:2]
                accelerator.backward(loss)
                optimizer.step()
                lrSchedule.step()
            progress_bar.update(1)
            total_train_loss += loss.item()
            if batch_idx % loss_log_steps == 0 and batch_idx > 0:
                train_loss_log.append((step, total_train_loss / loss_log_steps))
                total_train_loss = 0
            batch_idx += 1
            step += 1
            if time.time() > config["maxTime"]:
                break

        eval_data = lossEval(model, eval_dataloader, eval_data, config)

        if epoch % epoch_per_eval == 0 and epoch > 1:
            bleu_data = bleuEval(model, eval_dataloader, bleu_data, config)

        if epoch % epoch_per_save == 0 and epoch > 1:
            checkpoints = [int(check) for check in os.listdir(checkpointDir)]
            if len(checkpoints) >= max_num_checkpoints:
                checkpoints.sort()
                shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
            accelerator.save_state(output_dir=(checkpointDir + "/" + str(epoch)))

        if time.time() > config["maxTime"]:
            accelerator.print("Max Time exceeded: Stopped at " + str(datetime.timedelta(seconds=time.time() - config["startTime"])))
            break

    eval_data = lossEval(model, eval_dataloader, eval_data, config)

    bleu_data = bleuEval(model, eval_dataloader, bleu_data, config)

    checkpoints = [int(check) for check in os.listdir(checkpointDir)]
    checkpoints.sort()
    if len(checkpoints) >= max_num_checkpoints:
        shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
    accelerator.free_memory()
    unwrapped = accelerator.unwrap_model(model=model)
    accelerator.save(unwrapped.state_dict(), (outputDir + "/weights.pth"))

    save_results_to_json(train_loss_log, eval_data, bleu_data, config, (outputDir + "/results.json"))

def model_from_weights(model, weight_path):
    state = torch.load(weight_path)
    model.load_state_dict(state)
    return model

startTime = time.time()

config = {
    "append_language_tokens": False,
    "sampleFactor": {"total_new_data": {"lug": 1}},
    "eval_sampleFactor": {"total_new_data": {"lug": 1}},
    "dataSplitStep": 0,
    "training_epochs": 30,
    "output_dir": "lug_eng",
    "batch_size": 16,
    "maxTime": startTime + 120 * 60 * 60,
    "startTime": startTime,
    "LRschedule": "constant",
    "PBS_ID": "example_id"
}


dataset = load_dataset('csv', data_files={'train': '../train_eng_luganda.csv', 'test': '../test_eng_luganda.csv'}) # loads the dataset 
#  neeed a split 


dataset = create_dataset([dataset], ["total_new_data"], config)   # why do we turn it into a list ?

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-lg-en") #  loads the model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-lg-en") # loads tokenizer

train(model, tokenizer, dataset, config)
