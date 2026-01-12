"""
Test script to verify the model is compatible with HuggingFace's AutoModelForCausalLM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from model import GPT, GPTConfig
import tempfile
import os


def test_save_and_load():
    """Test saving and loading the model with HuggingFace API."""
    print("=" * 80)
    print("Testing HuggingFace compatibility...")
    print("=" * 80)
    
    # Create a small test model configuration
    config = GPTConfig(
        block_size=256,
        vocab_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=128,
        bias=True,
        n_exp=1,  # Start with dense model for simplicity
    )
    
    # Create model
    print("\n1. Creating model from config...")
    model = GPT(config)
    print(f"   ✓ Model created with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Create a temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model")
        
        # Save the model using HuggingFace API
        print(f"\n2. Saving model to {save_path}...")
        model.save_pretrained(save_path)
        print("   ✓ Model saved successfully")
        
        # Check that config was saved
        config_path = os.path.join(save_path, "config.json")
        if os.path.exists(config_path):
            print(f"   ✓ Config file created: {config_path}")
        else:
            print(f"   ✗ Config file not found!")
            return False
        
        # Load the model using AutoModelForCausalLM
        print("\n3. Loading model with AutoModelForCausalLM.from_pretrained()...")
        loaded_model = AutoModelForCausalLM.from_pretrained(save_path)
        print("   ✓ Model loaded successfully")
        
        # Verify it's the correct type
        print(f"\n4. Verifying model type...")
        print(f"   Original model type: {type(model).__name__}")
        print(f"   Loaded model type: {type(loaded_model).__name__}")
        assert isinstance(loaded_model, GPT), "Loaded model is not an instance of GPT"
        print("   ✓ Model type verified")
        
        # Test forward pass
        print("\n5. Testing forward pass...")
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Test with original model
        with torch.no_grad():
            output1 = model(input_ids=input_ids, return_dict=True)
            print(f"   Original model output shape: {output1.logits.shape}")
        
        # Test with loaded model
        with torch.no_grad():
            output2 = loaded_model(input_ids=input_ids, return_dict=True)
            print(f"   Loaded model output shape: {output2.logits.shape}")
        
        assert output1.logits.shape == output2.logits.shape
        print("   ✓ Forward pass successful")
        
        # Test backward compatibility with idx/targets
        print("\n6. Testing backward compatibility (idx/targets)...")
        with torch.no_grad():
            output3 = loaded_model(idx=input_ids, return_dict=False)
            logits, loss, losses = output3
            print(f"   Legacy output format works: logits={logits.shape}, loss={loss}, losses={losses}")
        print("   ✓ Backward compatibility verified")
        
    print("\n" + "=" * 80)
    print("✓ All tests passed! Model is compatible with HuggingFace AutoModel.")
    print("=" * 80)
    return True


def test_moe_model():
    """Test saving and loading an MoE model."""
    print("\n" + "=" * 80)
    print("Testing MoE model compatibility...")
    print("=" * 80)
    
    # Create a small MoE model configuration
    config = GPTConfig(
        block_size=256,
        vocab_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=128,
        bias=True,
        n_exp=8,  # 8 experts
        moe_top_k=2,  # Note: renamed from top_k to avoid HF generation conflict
        stride=2,  # MoE every 2 layers
        use_aux_loss=True,
    )
    
    print("\n1. Creating MoE model from config...")
    model = GPT(config)
    print(f"   ✓ MoE Model created with {model.get_num_params()/1e6:.2f}M parameters")
    print(f"   Experts: {config.n_exp}, Top-k: {config.moe_top_k}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_moe_model")
        
        print(f"\n2. Saving MoE model to {save_path}...")
        model.save_pretrained(save_path)
        print("   ✓ MoE Model saved successfully")
        
        print("\n3. Loading MoE model with AutoModelForCausalLM.from_pretrained()...")
        loaded_model = AutoModelForCausalLM.from_pretrained(save_path)
        print("   ✓ MoE Model loaded successfully")
        
        print("\n4. Testing MoE forward pass...")
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = loaded_model(input_ids=input_ids, labels=labels, return_dict=True)
            print(f"   Output shape: {output.logits.shape}")
            print(f"   Loss: {output.loss}")
        print("   ✓ MoE forward pass successful")
    
    print("\n" + "=" * 80)
    print("✓ MoE model tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        # Test basic model
        test_save_and_load()
        
        # Test MoE model
        test_moe_model()
        
        print("\n" + "=" * 80)
        print("SUCCESS! Your model is fully compatible with HuggingFace's API.")
        print("\nYou can now use:")
        print("  - model.save_pretrained('path/to/model')")
        print("  - AutoModelForCausalLM.from_pretrained('path/to/model')")
        print("  - model.push_to_hub('username/model-name')  # to share on HF Hub")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
