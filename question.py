from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")