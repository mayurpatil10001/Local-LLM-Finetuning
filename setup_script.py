#!/usr/bin/env python3
"""
Setup and Verification Script for Local LLM Training
Run this first to verify your system is ready
"""

import sys
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Detected: {gpu_name}")
            print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            
            # Recommend settings based on GPU memory
            if gpu_memory < 4:
                print("\n⚠ Warning: Low GPU memory. Use --batch-size 1")
            elif gpu_memory < 6:
                print("\n✓ Good GPU. Recommended: --batch-size 2")
            else:
                print("\n✓ Excellent GPU. Recommended: --batch-size 4")
            
            return True
        else:
            print("⚠ No GPU detected. Training will use CPU (much slower)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_packages():
    """Check required packages"""
    print_header("Checking Required Packages")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'rich': 'Rich',
        'accelerate': 'Accelerate'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_disk_space():
    """Check available disk space"""
    print_header("Checking Disk Space")
    
    import shutil
    total, used, free = shutil.disk_usage("/")
    
    free_gb = free / (2**30)
    print(f"Free space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("✗ Low disk space. At least 10GB recommended")
        return False
    elif free_gb < 20:
        print("⚠ Moderate disk space. 20GB+ recommended")
        return True
    else:
        print("✓ Sufficient disk space")
        return True

def install_packages(missing_packages):
    """Install missing packages"""
    print_header("Installing Missing Packages")
    
    print("This will install:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    response = input("\nProceed with installation? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled")
        return False
    
    try:
        # Install PyTorch with CUDA support
        if 'torch' in missing_packages:
            print("\nInstalling PyTorch with CUDA support...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            missing_packages.remove('torch')
        
        # Install other packages
        if missing_packages:
            print("\nInstalling other packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
        
        print("\n✓ All packages installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        return False

def create_sample_config():
    """Create sample configuration files"""
    print_header("Creating Sample Files")
    
    # Create sample questions file
    questions_file = Path("sample_questions.txt")
    if not questions_file.exists():
        with open(questions_file, 'w') as f:
            f.write("What is artificial intelligence?\n")
            f.write("How does machine learning work?\n")
            f.write("What is the difference between AI and ML?\n")
            f.write("Explain neural networks in simple terms.\n")
            f.write("What is deep learning?\n")
        print(f"✓ Created {questions_file}")
    
    # Create sample training script
    train_script = Path("train.sh")
    if not train_script.exists():
        with open(train_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Quick training script\n\n")
            f.write("python local_llm_training.py \\\n")
            f.write("    --model gpt2 \\\n")
            f.write("    --dataset eli5 \\\n")
            f.write("    --num-samples 5000 \\\n")
            f.write("    --epochs 2 \\\n")
            f.write("    --batch-size 4 \\\n")
            f.write("    --output-dir ./my_trained_model\n")
        train_script.chmod(0o755)
        print(f"✓ Created {train_script}")
    
    # Create inference script
    infer_script = Path("test_model.sh")
    if not infer_script.exists():
        with open(infer_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Test trained model\n\n")
            f.write("python inference_server.py \\\n")
            f.write("    --model-path ./my_trained_model/final_model \\\n")
            f.write("    --mode interactive\n")
        infer_script.chmod(0o755)
        print(f"✓ Created {infer_script}")
    
    print("\nSample files created. You can now:")
    print("  1. Run: ./train.sh (or bash train.sh)")
    print("  2. After training: ./test_model.sh")

def run_gpu_test():
    """Run a quick GPU test"""
    print_header("Running GPU Test")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("No GPU available for testing")
            return
        
        print("Testing GPU computation...")
        
        # Create test tensors
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Time matrix multiplication
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ GPU computation test passed ({elapsed*1000:.2f}ms)")
        print("GPU is working correctly!")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")

def print_recommendations():
    """Print training recommendations"""
    print_header("Training Recommendations")
    
    print("""
For HP Victus Laptop:

1. FIRST RUN (2-3 hours) - Test everything works:
   python local_llm_training.py --model gpt2 --dataset eli5 --num-samples 5000 --epochs 2

2. STANDARD RUN (6-7 hours) - Good quality:
   python local_llm_training.py --model gpt2 --dataset eli5 --num-samples 10000 --epochs 3

3. HIGH QUALITY (10-12 hours) - Best results:
   python local_llm_training.py --model gpt2-medium --dataset eli5 --num-samples 20000 --epochs 3

Tips:
- Keep laptop plugged in and on high performance mode
- Ensure good ventilation (use cooling pad if available)
- Monitor temperature: watch -n 1 nvidia-smi
- Training will auto-save every 500 steps
- Press Ctrl+C to stop safely (saves checkpoint)
- Resume with: --resume-from ./my_trained_model/checkpoints/checkpoint-XXXX

Datasets for Q&A:
- eli5: Best for conversational Q&A (Recommended)
- squad: Best for factual questions
- wikitext: Best for general knowledge
    """)

def main():
    """Main setup verification"""
    print("\n" + "="*60)
    print("  LOCAL LLM TRAINING - SETUP VERIFICATION")
    print("="*60)
    
    all_good = True
    
    # Check Python
    if not check_python_version():
        all_good = False
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check packages
    packages_ok, missing = check_packages()
    
    if not packages_ok:
        all_good = False
        install = input("\nInstall missing packages now? (y/n): ")
        if install.lower() == 'y':
            if install_packages(missing):
                packages_ok = True
                all_good = True
            else:
                all_good = False
    
    # Check disk space
    if not check_disk_space():
        print("\n⚠ Warning: Low disk space may cause issues")
    
    # Create sample files
    create_sample_config()
    
    # Run GPU test if available
    if gpu_available and packages_ok:
        run_gpu_test()
    
    # Final summary
    print_header("Setup Summary")
    
    if all_good:
        print("✓ Your system is ready for training!")
        print_recommendations()
        
        print("\n" + "="*60)
        print("QUICK START:")
        print("="*60)
        print("1. Run training: bash train.sh")
        print("2. Wait for training to complete (2-12 hours)")
        print("3. Test model: bash test_model.sh")
        print("="*60 + "\n")
    else:
        print("✗ Please fix the issues above before training")
        if not packages_ok:
            print("\nTo install packages manually:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("  pip install transformers datasets accelerate rich")

if __name__ == "__main__":
    main()