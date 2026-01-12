# HuggingFace Integration with trust_remote_code

This codebase has been refactored to support full HuggingFace compatibility, including `trust_remote_code` loading.

## File Structure

The model code uses HuggingFace-compatible modules:

- **`configuration_nanomoe_gpt.py`**: Contains `GPTConfig` (PretrainedConfig subclass)
- **`modeling_nanomoe_gpt.py`**: Contains `GPT` model, all components, and auto-registration
- **`manager.py`**: Required dependency for MoE loss tracking

## Loading Models

### Option 1: Local Import

For loading checkpoints in the same workspace:

```python
from modeling_nanomoe_gpt import GPT, GPTConfig
from transformers import AutoModelForCausalLM

# Importing modeling_nanomoe_gpt auto-registers the model
model = AutoModelForCausalLM.from_pretrained("./checkpoint_dir")
```

### Option 2: Remote Import with trust_remote_code (Portable)

For loading checkpoints from anywhere (e.g., HuggingFace Hub):

```python
from transformers import AutoModelForCausalLM

# No need to import model.py - loads code from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "./checkpoint_dir", 
    trust_remote_code=True
)
```

**Important**: With `trust_remote_code=True`, HuggingFace loads the model code from the checkpoint directory itself. The checkpoint must contain:
- `configuration_nanomoe_gpt.py`
- `modeling_nanomoe_gpt.py`
- `manager.py`
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)

## Saving Checkpoints

The training script ([train.py](train.py)) automatically includes all necessary files when saving:

```python
# In train.py (already implemented):
raw_model.save_pretrained(ckpt_dir)  # Saves weights + config.json

# Copy model code for trust_remote_code
import shutil
for filename in ['configuration_nanomoe_gpt.py', 'modeling_nanomoe_gpt.py', 'manager.py']:
    shutil.copy(filename, ckpt_dir)
```

## Checkpoint Structure

A complete checkpoint directory looks like:

```
checkpoint_dir/
├── config.json                      # Model configuration
├── model.safetensors                 # Model weights
├── configuration_nanomoe_gpt.py      # Config class (for trust_remote_code)
├── modeling_nanomoe_gpt.py           # Model class (for trust_remote_code)
└── manager.py                        # Dependencies (for trust_remote_code)
```

## Sharing on HuggingFace Hub

To share your model on HuggingFace Hub:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./checkpoint_dir")

# Upload to Hub (creates a new repository)
model.push_to_hub("your-username/nanomoe-gpt-model")
```

Then anyone can load it:

```python
model = AutoModelForCausalLM.from_pretrained(
    "your-username/nanomoe-gpt-model",
    trust_remote_code=True
)
```

## Example: Loading from Another Project

If you're in a different project (e.g., `nanochat`) and want to load a checkpoint:

```python
# In /home/lish/nanochat/scripts/base_eval.py

from transformers import AutoModelForCausalLM

# Load with trust_remote_code - no need to modify sys.path
model = AutoModelForCausalLM.from_pretrained(
    "/home/lish/nanoMoE/checkpoints/my-checkpoint",
    trust_remote_code=True  # Loads code from checkpoint directory
)
```

## Backward Compatibility

All existing code works by importing from the new modules:

```python
from modeling_nanomoe_gpt import GPT, GPTConfig

config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
model = GPT(config)

# Training with return_dict=False for tuple output
logits, loss, losses = model(input_ids=X, labels=Y, return_dict=False)
```

## Security Note

⚠️ `trust_remote_code=True` executes Python code from the checkpoint directory. Only use this with checkpoints you trust!

## Testing

Run the test suite to verify everything works:

```bash
# Test basic HuggingFace compatibility
python test_hf_compat.py

# Test trust_remote_code loading
python test_trust_remote_code.py
```
