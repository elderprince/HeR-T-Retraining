import re
import json
import torch
import numpy as np

from tqdm.auto import tqdm
from her_t_pytorch_lightning.dataset_loader import CustomImageDataset
from her_t_pytorch_lightning.utils import read_config
from transformers import DonutProcessor, VisionEncoderDecoderModel
from donut import JSONParseEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

kwargs = read_config('experiments/inference_config')

image_path = kwargs[['image_path']]
pretrained_processor = kwargs[['pretrained_processor']]
pretrained_model = kwargs[['pretrained_model']]
output_dir = kwargs[['output_dir']]

dataset = CustomImageDataset(image_path)
processor = DonutProcessor.from_pretrained(pretrained_processor)
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model)

model.eval()
model.to(device)

output_list, accs, files = [], [], []

for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    # Do some execption handling for wierd files
    try:
        # Load the image
        image = sample[0].convert("RGB")
        
        # Check if the image is truncated
        image.load()
    except OSError as e:
        if "image file is truncated" in str(e):
            print(f"Warning: Skipping truncated image")
            continue
        else:
            raise
                
    # prepare encoder inputs
    pixel_values = processor(sample[0].convert("RGB"), 
                             return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # prepare decoder inputs
    task_prompt = "<s_herbarium>"
    decoder_input_ids = processor.tokenizer(task_prompt, 
                                            add_special_tokens=False, 
                                            return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)
        
    # autoregressively generate sequence
    outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # turn into JSON
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, 
                      "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    seq = processor.token2json(seq)
    
    ground_truth = json.loads(sample[1])
    evaluator = JSONParseEvaluator()
    score = evaluator.cal_acc(seq, ground_truth)
    file = sample[2]

    accs.append(score)
    output_list.append(seq)
    files.append(file)

scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
print(f"Accuracy list length : {len(accs)}")
print("Mean accuracy: ", np.mean(accs))
print("Median accuracy:", np.median(accs))
print("Standard deviation: ", np.std(accs))

with open(output_file, 'w') as output_pre:
    for row in output_list:
        output_pre.write(str(row) + '\n')

with open(accuracy_file, 'w') as acc_pre:
    for row in accs:
        acc_pre.write(str(row) + '\n')

with open(filename_file, 'w') as file_names:
    for row in files:
        file_names.write(str(row) + '\n')