import torch, time, os
from datasets import load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration
from data_process_t import load_from_dict

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

max_input_length = 1024
batch_size = 8
trained_path = "o_pubmed_b4_g8_intro(1024)/checkpoint-37000"
start = time.time()
# load pubmed test dataset
pubmed_test_path = r"/home/ubuntu/ljl/dataset/original_dataset/pubmed-dataset/test.txt"
# pubmed_test_path = r"../dataset/pubmed-dataset/pubmed-dataset/test.txt"
# pubmed_test_path = r"extract/pubmed_test_extra.txt"
# pubmed_test_path = r"extract/arxiv_test_extra.txt"
pubmed_test = load_from_dict(pubmed_test_path)

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained(trained_path)
model = LEDForConditionalGeneration.from_pretrained(trained_path).to("cuda").half()

def generate_answer(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=max_input_length, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)

    return batch

result = pubmed_test.map(generate_answer, batched=True, batch_size=batch_size)

# load rouge
rouge = load_metric("transformers-master/src/transformers/data/metrics/rouge.py")

print("Test Results Rouge1:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rouge1"])['rouge1'].mid)
print("Test Results Rouge2:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rouge2"])['rouge2'].mid)
print("Test Results RougeL:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rougeL"])['rougeL'].mid)


end = time.time()
print("all the test using : ", str(end-start))
