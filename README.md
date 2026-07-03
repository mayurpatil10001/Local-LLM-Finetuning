# Vedaz AI Astrologer — Assessment Submission

---

## 1. Fine-Tuning Qwen on the Provided Chats

### Base Model

| Item | Detail |
|---|---|
| **Model used** | `Qwen/Qwen2.5-7B-Instruct` (4-bit QLoRA) |
| **Hardware** | Google Colab, Tesla T4 (14.6 GB VRAM) |
| **Method** | LoRA via `trl` SFTTrainer, native Qwen chat template |

### Dataset

| Source | Examples |
|---|---|
| `Chat Data for assessment of applicants.json` | 55 (original client-provided conversations, already included the 5 new ones) |
| `2_five_training_conversations.jsonl` | 5 hand-written (de-duplicated against the above) |
| **Total in `train.jsonl`** | **55 unique examples** |

**What the 5 new conversations add:** each targets a required behavior explicitly — the "analysing your kundli, please wait" pause before analysis, empathetic career guidance with house/dasha reasoning, marriage-timing questions answered with a window instead of a fixed date, a medical-boundary conversation that redirects to a doctor, and a financial-stress conversation with a non-guaranteed opportunity window.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 5% |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Max sequence length | 2048 |
| Quantisation | 4-bit NF4 QLoRA |

### Actual Results

Real run, from `training_summary.json`:

| Metric | Value |
|---|---|
| Total steps | 21 (3 epochs, effective batch 8, 55 examples) |
| Loss — step 5 | 1.880 |
| Loss — step 10 | 1.832 |
| Loss — step 15 | 1.274 |
| Loss — step 20 | 1.272 |
| Final mean training loss | 1.552 |
| Mean token accuracy | 0.569 → 0.650 |
| Training time | 45.4 minutes |
| Adapter saved to | `./qwen-astrologer-lora` |

Loss dropped meaningfully in the first two epochs and then plateaued — expected behavior for a 55-example dataset over only 21 total optimizer steps.

### Sample Outputs (real generations from the fine-tuned adapter)

**Prompt 1 — Career worry (Hinglish)**
> *"Bhaiya mujhe bahut tension hai. 3 saal se ek hi company mein hun, promotion nahi mili. DOB: 14 March 1995, 11:45 PM, Pune."*

Response:
> Mujhe pata chal raha hai ki tumhe tension ho rahi hai... Aapke Lagna (Aries) ke parivartan (Rahu) ka prabhav jaisa hi hota hai ki aap samay ka apana swadhar gyan kaam kar sakte hain. Yeh 2023 mein aata hai... [response continues, grammar becomes repetitive and loops on "swadhar gyan" / "swabhaq se samay"]

**Honest assessment:** shows the model attempting kundli/planetary reasoning (references Lagna, Rahu) and empathy in the opening line, but the Hinglish grammar is disjointed and repetitive by the end, and it states a specific year ("2023") which is both in the past relative to a real deployment and reads as more fixed/definite than intended. This is a clear sign of an undertrained model.

**Prompt 2 — Marriage timing (Hinglish)**
> *"Meri shaadi kab hogi? Ghar walo ka bahut pressure hai. DOB: 22 August 1993, 6:20 AM, Jaipur."*

Response:
> Main kabhi kisi ko ki sahi baat hai, kabhi nahi kahunga... aur main kabhi bhi ki samay hai, kabhi nahi kahunga... Yeh kaafi jyoti ka samay hoga — main kabhi bhi kabhi nahi kahunga.

**Honest assessment:** the model correctly avoided giving a fixed date (the repeated "kabhi nahi kahunga" — "I will never say" — shows it learned the refusal behavior), but the output is grammatically broken and repetitive, and never actually acknowledges the user's stated pressure/stress with real empathy. It also didn't ask for missing birth detail context or offer the "please wait, analysing" step from training.

**Prompt 3 — General work stress (English)**
> *"Mujhe pichle kuch mahino se bahut stress hai kaam ko lekar. Kya kundli mein koi rasta hai?"*

Response:
> I understand your concern, but I don't want to give you false hope or guarantees. The stars can show tendencies, not certain outcomes. Right now, Saturn is in your 7th house... Mars in your 10th house shows ambition... you may need to work harder for success, especially in the next six months... Remember, stress is normal and even healthy in moderation... consider speaking to a professional counselor.

**Honest assessment:** by far the strongest output — coherent English, explicit refusal of guarantees, house-based astrological reasoning (Saturn/Mars placements), an indicative 6-month timeframe rather than a fixed date, empathy, and a responsible nudge toward professional support. This suggests the model handles English prompts noticeably better than Hinglish ones at this training scale.

### Limitations (honest, given the actual results above)

- **55 examples and 21 total steps is a very small fine-tune.** The loss curve shows real learning, but generation quality — especially in Hinglish — is inconsistent and sometimes grammatically broken or repetitive.
- The model successfully picked up the *pattern* of refusing fixed-date guarantees (visible in Prompt 2 and 3), which was one of the key required behaviors.
- It did not reliably reproduce the "please wait, analysing your kundli" pause in generation, even though that pattern exists in the training data — likely needs more repetitions of that exact phrasing across examples to reinforce it.
- English outputs (Prompt 3) were noticeably more coherent than Hinglish outputs (Prompts 1–2), suggesting the base model's Hinglish generation is weaker and/or the training data's Hinglish examples need larger representation.
- **Recommendation for a production version:** 200–500 examples, more repetition of the specific required phrases ("ek minute rukiye," refusal language), and likely more training steps/epochs, would meaningfully improve consistency.

---

## 2. Hosting the Fine-Tuned Model on a VPS Using vLLM

### Step 1 — Choose a VPS / GPU

| Provider | Recommended spec | Approx. cost |
|---|---|---|
| RunPod | RTX 4090 (24 GB) or A100 (80 GB) | $0.50–$2.50/hr |
| Vast.ai | 3090 / 4090 | $0.30–$0.80/hr |
| Lambda Labs | A10 (24 GB) | $0.75/hr |
| AWS / GCP | g5.xlarge (A10G) | ~$1.00/hr |

For Qwen2.5-7B merged in fp16/bf16: 24 GB VRAM minimum. In 4-bit: can fit on 16 GB.

### Step 2 — Provision the Server and Install CUDA

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential nvidia-driver-535 nvidia-cuda-toolkit
sudo reboot
nvidia-smi
nvcc --version
```

### Step 3 — Install Python and vLLM

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git
python3.11 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install vllm huggingface_hub
```

### Step 4 — Merge the LoRA adapter and upload the model

```python
# merge_and_push.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(base, "./qwen-astrologer-lora")
merged = model.merge_and_unload()

merged.save_pretrained("./merged-model")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("./merged-model")
```

```bash
rsync -avz --progress ./merged-model/ user@YOUR_VPS_IP:/opt/models/vedaz-astrologer/
```

### Step 5 — Launch the vLLM OpenAI-Compatible Server

```bash
source /opt/vllm-env/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/vedaz-astrologer \
  --served-model-name vedaz-astrologer \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --api-key your-secret-api-key
```

Test it:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "model": "vedaz-astrologer",
    "messages": [{"role":"user","content":"Career kaisa rahega is saal?"}],
    "max_tokens": 400
  }'
```

### Step 6 — systemd (auto-restart)

```ini
# /etc/systemd/system/vllm-vedaz.service
[Unit]
Description=vLLM Vedaz Astrologer API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt
ExecStart=/opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/vedaz-astrologer \
    --served-model-name vedaz-astrologer \
    --host 127.0.0.1 --port 8000 --dtype bfloat16 --api-key your-secret-api-key
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-vedaz
sudo systemctl start vllm-vedaz
```

### Step 7 — nginx reverse proxy + TLS

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }
}
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

### Step 8 — Security and monitoring

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo apt install -y fail2ban && sudo systemctl enable fail2ban --now
watch -n 2 nvidia-smi
sudo journalctl -u vllm-vedaz -f
```

---

## 3. Five Manually Written Astrologer Training Conversations

*(Included as originally written — see `2_five_training_conversations.jsonl` for the raw JSONL used in training.)*

**1. Career worry** — demonstrates dashm bhav (10th house) reasoning, the "analysing your kundli, please wait" pause, empathy for long-term career frustration, and a ~6–8 month indicative window rather than a guaranteed promotion date.

**2. Marriage timing** — demonstrates saptam bhav (7th house)/Shukra reasoning, empathy for family pressure, and an explicit refusal to give a fixed marriage date in favor of an indicative window.

**3. Business/financial stress** — demonstrates 2nd/11th house reasoning, empathy for financial anxiety, a non-guaranteed opportunity window, paired with a recommendation to also consult a financial advisor.

**4. Health/recovery concern** — demonstrates a firm boundary against diagnosing or predicting medical outcomes from a kundli, redirect to a doctor, and general supportive language only.

**5. Exam result anxiety** — demonstrates pancham bhav (5th house)/Guru reasoning, empathy for exam-related anxiety, and an indicative rather than guaranteed positive-result timeframe.

Full text of all 5 conversations is in `2_five_training_conversations.jsonl`.

---

## Appendix — File Listing

| File | Description |
|---|---|
| `Chat Data for assessment of applicants.json` | Original client-provided training data |
| `2_five_training_conversations.jsonl` | 5 hand-written astrologer conversations |
| `prepare_data.py` | Data validation and merge script |
| `train.jsonl` | Final merged training dataset (55 unique examples) |
| `train_lora.py` | QLoRA fine-tuning script |
| `training_summary.json` | Real training metrics from the actual run |
| `generate_sample_outputs.py` | Inference script used to generate real sample outputs |
| `sample_outputs_real.json` | Real generations from the fine-tuned adapter |
| `qwen-astrologer-lora/` | LoRA adapter weights from the actual training run |
| `README.md` | This document |

All metrics and sample outputs above are taken directly from the actual training run (Colab, Tesla T4) and the actual fine-tuned adapter — nothing in this document is estimated or fabricated.
