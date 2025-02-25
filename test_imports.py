print("Testing imports one by one:")
print("-" * 50)

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully (version {tf.__version__})")
except Exception as e:
    print(f"✗ TensorFlow import failed: {str(e)}")

# Test ONNX
try:
    import onnx
    print(f"✓ ONNX imported successfully (version {onnx.__version__})")
except Exception as e:
    print(f"✗ ONNX import failed: {str(e)}")

# Test onnx-tf
try:
    from onnx_tf.backend import prepare
    print("✓ onnx-tf imported successfully")
except Exception as e:
    print(f"✗ onnx-tf import failed: {str(e)}")

# Test TensorFlow Probability
try:
    import tensorflow_probability as tfp
    print(f"✓ TensorFlow Probability imported successfully (version {tfp.__version__})")
except Exception as e:
    print(f"✗ TensorFlow Probability import failed: {str(e)}")

print("-" * 50)
