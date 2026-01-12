"""
Test script to verify trust_remote_code loading works correctly.
This creates a temporary checkpoint and loads it back using trust_remote_code=True.
"""

import tempfile
import shutil
import os
from modeling_nanomoe_gpt import GPT, GPTConfig
from transformers import AutoModelForCausalLM

def test_trust_remote_code():
    # Create a small model
    config = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        bias=False,
        n_exp=1,  # Start with dense model
    )
    model = GPT(config)
    
    # Create temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, 'test_checkpoint')
        
        # Save the model
        print(f"Saving model to {ckpt_dir}...")
        model.save_pretrained(ckpt_dir)
        
        # Copy the necessary files for trust_remote_code
        for filename in ['configuration_nanomoe_gpt.py', 'modeling_nanomoe_gpt.py', 'manager.py']:
            src = filename
            dst = os.path.join(ckpt_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
        print("Copied model files for trust_remote_code loading")
        
        # Try loading with trust_remote_code=True
        print(f"Loading model with trust_remote_code=True...")
        loaded_model = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True)
        
        print("✓ Successfully loaded model with trust_remote_code=True!")
        print(f"Model type: {type(loaded_model)}")
        print(f"Config type: {type(loaded_model.config)}")
        
        return True

if __name__ == "__main__":
    try:
        test_trust_remote_code()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
