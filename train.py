import time, torch, os
from datasets import load_metric
from data_process_t import load_from_dict
from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# set the parameters
encoder_max_length = 512+1024
decoder_max_length = 512
batch_size = 8
gradient_accumulation_steps = 4
train_eopch = 10
pretrained_model_path = "led_base_16384"
trained_model_saved_path = "o_pubmed_b4_g8_extract(512)_intro(1024)"

start = time.time()
# load_rouge
rouge = load_metric("transformers-master/src/transformers/data/metrics/rouge.py")

# load pubmed train and eval
pubmed_train_path = r"/home/ubuntu/ljl/dataset/all_with_extract/all_pubmed_extract_train.txt"
pubmed_val_path = r"/home/ubuntu/ljl/dataset/all_with_extract/all_pubmed_extract_val.txt"

pubmed_train = load_from_dict(pubmed_train_path)
pubmed_val = load_from_dict(pubmed_val_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

def process_data_to_model_inputs(batch):
	# tokenize the inputs and labels
	inputs = tokenizer(
		batch["article"],
		padding="max_length",
		truncation=True,
		max_length=encoder_max_length,
	)

	outputs = tokenizer(
		batch["abstract"],
		padding="max_length",
		truncation=True,
		max_length=decoder_max_length,
	)
	batch["input_ids"] = inputs.input_ids
	batch["attention_mask"] = inputs.attention_mask

	# create 0 global_attention_mask lists
	batch["global_attention_mask"] = len(batch["input_ids"]) * [[0 for _ in range(len(batch["input_ids"][0]))]]

	# since above lists are references, the following line changes the 0 index for
	batch["global_attention_mask"][0][0] = 1
	batch["labels"] = outputs.input_ids

	# we have to make sure that the PAD token is ignored
	batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
					   batch["labels"]]

	return batch

# map train data
pubmed_train = pubmed_train.map(
	process_data_to_model_inputs,
	batched=True,
	batch_size=batch_size,
	remove_columns=["article", "abstract", "section_names"],
)

# map val data
pubmed_val = pubmed_val.map(
	process_data_to_model_inputs,
	batched=True,
	batch_size=batch_size,
	remove_columns=["article", "abstract", "section_names"],
)

# set python list to pytorch list
pubmed_train.set_format(
	type="torch",
	columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
pubmed_val.set_format(
	type="torch",
	columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
	predict_with_generate=True,
	evaluation_strategy="steps",
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	fp16=True,
	# fp16_backend="apex",
	output_dir=trained_model_saved_path,
	logging_steps=250,
	eval_steps=5000,
	save_steps=500,
	warmup_steps=1500,
	save_total_limit=2,
	gradient_accumulation_steps=gradient_accumulation_steps,
	num_train_epochs=train_eopch,
)
# num_train_epochs=1, not set here, so where to set

def compute_metrics(pred):
	labels_ids = pred.label_ids
	pred_ids = pred.predictions

	pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
	labels_ids[labels_ids == -100] = tokenizer.pad_token_id
	labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

	rouge_output1 = rouge.compute(
		predictions=pred_str, references=labels_str, rouge_types=["rouge1"]
	)["rouge1"].mid

	rouge_output2 = rouge.compute(
		predictions=pred_str, references=labels_str, rouge_types=["rouge2"]
	)["rouge2"].mid

	rouge_outputL = rouge.compute(
		predictions=pred_str, references=labels_str, rouge_types=["rougeL"]
	)["rougeL"].mid

	return {
		"rouge1-p,r,f": [round(rouge_output1.precision, 4),
						 round(rouge_output1.recall, 4),
						 round(rouge_output1.fmeasure, 4)],
		"rouge2-p,r,f": [round(rouge_output2.precision, 4),
						 round(rouge_output2.recall, 4),
						 round(rouge_output2.fmeasure, 4)],
		"rougeL-p,r,f": [round(rouge_outputL.precision, 4),
						 round(rouge_outputL.recall, 4),
						 round(rouge_outputL.fmeasure, 4)]
	}

# load_model + enable gradient checkpointing & disable cache for checkpointing
# not complete here and question
# torch.cuda.set_device(2)
led = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path, gradient_checkpointing=True, use_cache=False)

led.config.num_beams = 4
led.config.max_length = 512
led.config.mini_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.no_repeat_ngram_size = 3

# net = torch.nn.DataParallel(led, device_ids=device_ids)
# if torch.cuda.device_count()>2:
# 	led = torch.nn.DataParallel(led, device_ids=device_ids)
# 	print("dataparallel to device 0 and 1")
# 	led = led.cuda(device=device_ids)
# set generate hyperparameters

trainer = Seq2SeqTrainer(
	model=led,
	tokenizer=tokenizer,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=pubmed_train,
	eval_dataset=pubmed_val,
)

trainer.train()
end = time.time()
print("all the train using : ", str(end-start))
