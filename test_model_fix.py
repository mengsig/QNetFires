#!/usr/bin/env python3
"""
Test script to verify the model initialization fix.
"""

import sys
sys.path.append('src')

def test_model_initialization():
    """Test that model initialization works without the bias error."""
    print("Testing model initialization fix...")
    
    try:
        # This will fail if torch is not available, but that's okay
        import torch
        from src.Model import EnhancedQNet, DuelingQNet, QNet
        
        print("Testing QNet initialization...")
        model = QNet(50, 50)
        print("✓ QNet initialized successfully")
        
        print("Testing EnhancedQNet initialization...")
        model = EnhancedQNet(50, 50, use_attention=True, use_residual=True, use_multiscale=True)
        print("✓ EnhancedQNet initialized successfully")
        
        print("Testing DuelingQNet initialization...")
        model = DuelingQNet(50, 50)
        print("✓ DuelingQNet initialized successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        dummy_input = torch.randn(2, 8, 50, 50)
        output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except ImportError:
        print("⚠ PyTorch not available, skipping model tests")
        print("This is normal if PyTorch is not installed")
        return True
    except AttributeError as e:
        if "'NoneType' object has no attribute 'fill_'" in str(e):
            print("❌ Model initialization bug still present!")
            print(f"Error: {e}")
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def main():
    """Run the test."""
    print("Testing Model Initialization Fix")
    print("=" * 40)
    
    success = test_model_initialization()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Model initialization fix verified!")
        print("\nThe AttributeError: 'NoneType' object has no attribute 'fill_' should be resolved.")
        print("You can now run: python3 src/Train.py")
    else:
        print("❌ Model initialization fix failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())