import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import Runnable
import os


class Gemma3(Runnable):
    def __init__(self):
        ckpt = "google/gemma-3-1b-pt"
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = Gemma3ForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        print('Gemma3 LLM Loaded..')

    def invoke(self, prompt, config=None):
        
        response = self.llm.invoke(prompt, config=config)

        return response


class Mistral(Runnable):
    def __init__(self):
        ckpt = "./Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=os.environ.get("HF_HOME", "/tmp/offload"),
        )
        print('✅ Mistral-7B-Instruct (4-bit) loaded on GPU.')
    
    def invoke(self, prompt, config=None):
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])