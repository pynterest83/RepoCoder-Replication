import torch
import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import Tools

class CodeGen:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        print(f'Loaded model {model_name} with {self.model.num_parameters()} parameters')

    def generate(self, prompts, max_new_tokens=100):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids, attention_mask = inputs['input_ids'].cuda(), inputs['attention_mask'].cuda()

        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample = False,
                max_new_tokens = max_new_tokens
            )

        # batch_size x seq_len
        gen_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        # remove the prompt
        gen_texts = [
            gen_text[len(prompt):].strip() for gen_text, prompt in zip(gen_texts, prompts)
        ]
        return gen_texts

    def generate_by_batch_size (self, input_file, output_file, max_new_tokens=100):
        print(f'Generating code for {input_file}...')
        lines = Tools.load_jsonl(input_file)
        prompts = [f"{line['prompt']}\n" for line in lines]

        generated_codes = []
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch_prompts = prompts[i:i+self.batch_size]
            generated_code = self.generate(batch_prompts, max_new_tokens=max_new_tokens)
            generated_codes.extend(generated_code)
        
        if len(generated_codes) != len(lines):
            raise ValueError('Number of generated codes does not match number of prompts')
        
        new_lines = [
            {
                'prompt': line['prompt'],
                'metadata': line.get('metadata', {}),
                'choices': [{'text': generated_code}]
            }
            for line, generated_code in zip(lines, generated_codes)
        ]
        Tools.dump_jsonl(new_lines, output_file)
        print(f'Generated code saved to {output_file}')

if __name__ == '__main__':
    model = 'Salesforce/codegen-350M-multi'
    file_path = 'prompts/rg-one-gram-ws-20-ss-2.jsonl'
    output_path = 'predictions/' + file_path.split('/')[-1].replace('.jsonl', '_') + model.split('/')[-1] + '.jsonl'
    print(output_path)

    cg = CodeGen(model, batch_size=1)
    cg.generate_by_batch_size(file_path, output_path, max_new_tokens=100)