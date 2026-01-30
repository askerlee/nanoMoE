from transformers import AutoTokenizer, AutoModelForCausalLM

mname="nanoMoE_out0/sb2math-1000"
tok = AutoTokenizer.from_pretrained(mname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mname, trust_remote_code=True, device_map="auto")
tok.pad_token = tok.eos_token
model.generation_config.pad_token_id = tok.pad_token_id

# Try a manual generate
prompt = "Problem: Let $\\mathbf{a}$ and $\\mathbf{b}$ be vectors such that\n\\[\\mathbf{v} = \\operatorname{proj}_{\\mathbf{a}} \\mathbf{v} + \\operatorname{proj}_{\\mathbf{b}} \\mathbf{v}\\]for all vectors $\\mathbf{v}.$  Enter all possible values of $\\mathbf{a} \\cdot \\mathbf{b},$ separated by commas.\nAnswer:"

inp = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False))
