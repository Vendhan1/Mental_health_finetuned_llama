from datasets import load_dataset
from colorama import Fore
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import SFTTrainer , SFTConfig
from peft import LoraConfig , prepare_model_for_kbit_training
import torch


dataset = load_dataset('json',data_files='instructions.json',split='train')
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

def format_chat_template(batch,tokenizer):
    system_prompt= """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    samples =[]
    questions= batch["question"]
    answers=batch["answer"]
    for i in range(len(questions)):
        row_json=[
            {'role':'system',"content": system_prompt},
            {"role":"user","content":questions[i]},
            {"role":"assistant","content":answers[i]}
        ]
        text=tokenizer.apply_chat_template(row_json,tokenize=False)
        samples.append(text)
    return {
        "instruction":questions,
        "response":answers,
        "text":samples
    }
    
if __name__ == "__main__":
    base_model = "meta-llama/Llama-3.2-1B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token="ACCESS_TOKEN"
    )

    dataset = load_dataset('json', data_files='instructions.json', split='train')
    print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

    # Multiprocessing-safe mapping
    train_dataset = dataset.map(
        lambda x: format_chat_template( 
            x,
            AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                trust_remote_code=True,
                token="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"
            )
        ),
        num_proc= 8,
        batched=True,
        batch_size=8
    )

    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)
    
    
    model= AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        token="ACCESS_TOKEN",
        cache_dir="./workspace"
    )
    print(Fore.CYAN + str(model) + Fore.RESET)
    print(Fore.LIGHTYELLOW_EX+ str(next(model.parameters())))
    
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config=LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )
    
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B",num_train_epochs=3,per_device_train_batch_size=1,fp16=True),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")
