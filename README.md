# Local LLM Training 

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv llm_env
source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate rich
```

### 2. Train Your Model

**Option A: Quick Training (Recommended for first try - 2-3 hours)**
```bash
python local_llm_training.py \
    --model gpt2 \
    --dataset wikitext \
    --num-samples 5000 \
    --epochs 2 \
    --batch-size 4 \
    --output-dir ./my_trained_model
```

**Option B: Better Quality (6-7 hours)**
```bash
python local_llm_training.py \
    --model gpt2 \
    --dataset eli5 \
    --num-samples 10000 \
    --epochs 3 \
    --batch-size 4 \
    --output-dir ./my_trained_model
```

**Option C: High Quality (10-12 hours)**
```bash
python local_llm_training.py \
    --model gpt2-medium \
    --dataset eli5 \
    --num-samples 20000 \
    --epochs 3 \
    --batch-size 2 \
    --output-dir ./my_trained_model
```

**Option D: Extended Training (10,000 steps - 8-10 hours)**
```bash
# Set environment variable to improve CUDA memory allocation
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start training with lower batch size for stability
python local_llm_training.py \
    --model gpt2 \
    --dataset wikitext \
    --num-samples 10000 \
    --epochs 4 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --max-length 192 \
    --output-dir ./my_trained_model_10k_steps
```

## Training Parameters
- `--model` (string, default `gpt2`): Base model to fine-tune.
- `--dataset` (choices: `eli5`, `squad`, `wikitext`, `custom`): Dataset to train on.
- `--num-samples` (int, default `5000`): Number of effective samples used for training.
- `--epochs` (int, default `2`): Number of training epochs.
- `--batch-size` (int, default `4`): Per-device train batch size.
- `--gradient-accumulation-steps` (int, default `1`): Accumulate gradients to simulate larger batches.
- `--max-length` (int, default `256`): Max tokenized sequence length.
- `--learning-rate` (float, default `5e-5`): Optimizer learning rate.
- `--resume-from` (path, optional): Resume training from a specific checkpoint.
- `--output-dir` (path, default `./my_trained_model`): Where to save checkpoints and final model.
- `--custom-data-file` (path, optional): Text file for `--dataset custom` (one example per line).

Notes
- Windows optimizations enabled: `dataloader_pin_memory=False`, `dataloader_num_workers=0` to reduce memory pressure.
- Automatic OOM recovery: reduces `--batch-size` and resumes from latest checkpoint; if needed, reduces `--max-length`.
- Recommended on Windows/NVIDIA: set `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to reduce CUDA memory fragmentation.

### 3. Resume Training if Interrupted

```bash
# Find the latest checkpoint
python local_llm_training.py \
    --model gpt2 \
    --dataset wikitext \
    --resume-from ./my_trained_model/checkpoints/checkpoint-1000 \
    --output-dir ./my_trained_model
```

**Automatic OOM Recovery**
The training script now includes automatic CUDA Out-of-Memory (OOM) recovery:
```bash
# Training will automatically:
# 1. Reduce batch size if OOM occurs
# 2. Resume from latest checkpoint
# 3. Reduce sequence length if needed
# 4. Continue training without manual intervention
```

### 4. Test Your Model

**Interactive Chat Mode:**
```bash
python inference_server.py \
    --model-path ./my_trained_model/final_model \
    --mode interactive
```

**Single Question:**
```bash
python inference_server.py \
    --model-path ./my_trained_model/final_model \
    --mode single \
    --query "What is the capital of France?"
```

**Testing the 10k Steps Model:**
```bash
# Single question test
python inference_server.py \
    --model-path ./my_trained_model_10k_steps/final_model \
    --mode single \
    --query "What is artificial intelligence?"

# Interactive mode
python inference_server.py \
    --model-path ./my_trained_model_10k_steps/final_model \
    --mode interactive
```

**Batch Processing:**
```bash
# Create questions.txt with one question per line
python inference_server.py \
    --model-path ./my_trained_model/final_model \
    --mode batch \
    --questions-file questions.txt \
    --output-file answers.json
```

## Inference Commands and Parameters
- `--model-path` (path, required): Path to the trained model directory (`final_model`).
- `--mode` (choices: `interactive`, `batch`, `single`): Inference mode to use.
- `--query` (string, required for `single`): The single question to ask.
- `--questions-file` (path, required for `batch`): Text file with one question per line.
- `--output-file` (path, optional for `batch`, default `answers.json`): Where batch results are saved.

Behavior
- Device auto-detection: uses GPU if available, otherwise CPU.
- Output: answers are printed to console in `single` and `interactive`; saved to JSON in `batch`.
- Example batch file: use `sample_questions.txt` or create your own `questions.txt`.

## 📊 Recommended Datasets

### For Basic Q&A:

1. **eli5** (Explain Like I'm 5) - Best for Q&A
   - Questions with simple explanations
   - Great for conversational responses
   - Size: ~270k examples

2. **squad** (Stanford Question Answering)
   - High-quality question-answer pairs
   - Factual answers
   - Size: ~100k examples

3. **wikitext** (Wikipedia Text)
   - General knowledge
   - Good for factual information
   - Size: 103M tokens (wikitext-103)

### Usage:
```bash
# ELI5 (Best for beginners)
--dataset eli5

# SQuAD (Best for factual Q&A)
--dataset squad

# WikiText (Best for general knowledge)
--dataset wikitext
```

## 🔧 Optimizing for Your HP Victus

### Check Your GPU:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Adjust Settings Based on GPU Memory:

**If you have 4GB GPU:**
```bash
--model gpt2 --batch-size 2 --num-samples 5000
```

**If you have 6GB GPU:**
```bash
--model gpt2 --batch-size 4 --num-samples 10000
```

**If you have 8GB+ GPU:**
```bash
--model gpt2-medium --batch-size 4 --num-samples 20000
```

## 💾 Checkpoint System

The training automatically saves checkpoints:
- Every 500 steps
- On keyboard interrupt (Ctrl+C)
- At the end of training
- Keeps last 5 checkpoints to save disk space

Checkpoints are saved in: `./my_trained_model/checkpoints/`

## ⚡ Tips for Best Results

1. **Start Small**: Begin with 5000 samples and 2 epochs to test
2. **Monitor GPU**: Watch GPU temperature (use `nvidia-smi`)
3. **Use Good Cooling**: Ensure laptop has proper ventilation
4. **Power Settings**: Keep laptop plugged in and set to high performance
5. **Background Apps**: Close unnecessary applications
6. **Dataset Choice**: Use `eli5` for best Q&A performance

## 🐛 Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size
--batch-size 2

# Use smaller model
--model distilgpt2

# Reduce samples
--num-samples 3000
```

### Slow Training:
- Verify GPU is being used (check logs)
- Reduce num-samples for faster iteration
- Use smaller max-length: `--max-length 256`

### Model Not Answering Well:
- Train for more epochs: `--epochs 5`
- Use more samples: `--num-samples 20000`
- Try different dataset: `--dataset eli5`

## 📁 File Structure After Training

```
my_trained_model/
├── checkpoints/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-1500/
├── final_model/          # Use this for inference
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── logs/
└── training_config.json
```

## 🎯 Expected Training Times (HP Victus)

Assuming RTX 3050/3060 GPU:

| Configuration | Samples | Time | Quality |
|--------------|---------|------|---------|
| Quick Test   | 5,000   | 2-3h | Basic   |
| Standard     | 10,000  | 6-7h | Good    |
| High Quality | 20,000  | 10-12h | Better |

## 🔄 Example Workflow

```bash
# Day 1: Quick test (2-3 hours)
python local_llm_training.py --model gpt2 --dataset eli5 --num-samples 5000 --epochs 2

# Test the model
python inference_server.py --model-path ./my_trained_model/final_model --mode interactive

# Day 2: If satisfied, train longer (6-7 hours)
rm -rf ./my_trained_model  # Clean previous training
python local_llm_training.py --model gpt2 --dataset eli5 --num-samples 15000 --epochs 3
```

## 📝 Creating Custom Dataset

If you want to use your own data:

```python
# Create a file: my_data.txt
# One text per line or paragraph

# Then modify the script to load it:
from datasets import Dataset

with open('my_data.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({"text": texts})
```

## 🎉 Success Indicators

Your model is working well if it:
- Responds coherently to questions
- Doesn't repeat the question
- Provides relevant answers
- Completes sentences properly

Start with the quick test to verify everything works, then scale up!
