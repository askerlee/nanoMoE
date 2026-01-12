## HuggingFace Integration

The model has been refactored to be fully compatible with HuggingFace's Transformers library. You can now use `AutoModelForCausalLM.from_pretrained()` to load models.

### Quick Start

#### Saving a Model

```python
from model import GPT, GPTConfig

# Create and train your model
config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    vocab_size=50304,
)
model = GPT(config)

# ... train your model ...

# Save with HuggingFace API
model.save_pretrained("path/to/save/directory")
```

#### Loading a Model

```python
from transformers import AutoModelForCausalLM

# Load using AutoModel - no need to import GPT!
model = AutoModelForCausalLM.from_pretrained("path/to/save/directory")

# Or load from HuggingFace Hub (after pushing)
model = AutoModelForCausalLM.from_pretrained("username/model-name")
```

### Configuration Changes

**Important:** The `top_k` parameter in `GPTConfig` has been renamed to `moe_top_k` to avoid conflicts with HuggingFace's generation `top_k` parameter.

#### Old Code (Before Refactoring)
```python
config = GPTConfig(
    n_exp=8,
    top_k=2,  # Old name
)
```

#### New Code (After Refactoring)
```python
config = GPTConfig(
    n_exp=8,
    moe_top_k=2,  # New name to avoid HF conflict
)
```

**Note:** Internally, the config still stores this as `config.top_k` for backward compatibility with the model code, so existing training scripts using `config.top_k` will still work.

### MoE Model Example

```python
from model import GPT, GPTConfig

# Create MoE configuration
config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_exp=8,        # 8 experts
    moe_top_k=2,    # Top-2 routing
    stride=2,       # MoE every 2 layers
    use_aux_loss=True,
)

model = GPT(config)

# Save and load with HuggingFace
model.save_pretrained("my_moe_model")
loaded_model = AutoModelForCausalLM.from_pretrained("my_moe_model")
```

### Sharing on HuggingFace Hub

```python
# Login to HuggingFace (one time setup)
from huggingface_hub import login
login()

# Push to Hub
model.push_to_hub("username/model-name")

# Anyone can then load it
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("username/model-name")
```

### Forward Pass API

The forward method now supports both HuggingFace's standard arguments and the original arguments:

#### HuggingFace Style (Recommended)
```python
outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
loss = outputs.loss
logits = outputs.logits
```

#### Legacy Style (Still Supported)
```python
logits, loss, losses = model(idx=input_ids, targets=labels, return_dict=False)
```

### Generation

The model now supports HuggingFace's `generate()` method:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/model")
# Note: You'll need to save/load a tokenizer separately

input_ids = ...  # your tokenized input
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

For the legacy generation method:
```python
outputs = model.generate_legacy(
    idx=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)
```

### Backward Compatibility

All existing training scripts should continue to work with minimal changes:

1. If you're creating a new model, use `moe_top_k` instead of `top_k` in the config
2. Existing scripts that use `config.top_k` will still work (it's stored internally as `top_k`)
3. The old `from_pretrained()` method has been renamed to `from_pretrained_gpt2()` for loading official GPT-2 checkpoints

Example:
```python
# Old way (still works)
model = GPT.from_pretrained_gpt2('gpt2')

# New way for custom models
model = AutoModelForCausalLM.from_pretrained('path/to/model')
```

### Running Tests

To verify the integration works correctly:

```bash
python test_hf_compat.py
```

This will test both dense and MoE models with save/load functionality.
