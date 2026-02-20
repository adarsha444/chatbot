Once you successfully execute

```
docker compose up -d
```

Wait for around 5 minutes, till it finishes pulling the model. You can inspect logs to see progress if you want with command

```
docker logs -f ollama
```

To remove current model and try new one

```
docker compose down -v
```

and on docker-compose update and pull and try new one

once you find valid model and would like to use that moving forward just stop it at times you are not using it with

```
docker compose stop
```

To restart just use
```
docker compose start
```

### To Use GPU

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

and start with another docker file

```
docker compose -f docker-compose-gpu.yaml up -d
```

# Ollama Model Naming Guide

## Understanding Model Names

Ollama model names follow this pattern:

```
model-family:size-variant-quantization
```

**Example:** `llama3:8b-instruct-q4_0`

Breaking it down:
- **llama3** = Model family (base architecture)
- **8b** = 8 billion parameters
- **instruct** = Fine-tuning type
- **q4_0** = Quantization level

## Model Families

| Family | Creator | Best For |
|--------|---------|----------|
| **llama3** | Meta | General purpose, coding, reasoning |
| **qwen** | Alibaba | Multilingual, especially Asian languages |
| **mistral** | Mistral AI | Efficient reasoning, good quality-to-size ratio |
| **phi** | Microsoft | Extremely small but capable models |
| **gemma** | Google | Efficient, good for limited resources |
| **deepseek** | DeepSeek | Coding, technical tasks |

## Parameter Sizes

The "b" means billion parameters. Bigger = better quality but more resources:

- **1-3b**: Minimal resources, basic tasks
- **7-8b**: Good balance for most uses
- **13-14b**: Better reasoning
- **70b+**: Best quality (requires significant resources)

## Fine-tuning Variants

### instruct
**Example:** `llama3:8b-instruct`
- Trained to follow instructions and chat
- **Use this for chatbots and assistants**

### chat
**Example:** `qwen:7b-chat`
- Optimized for multi-turn conversations
- Similar to instruct, slightly different training

### base
**Example:** `llama3:8b` (no suffix)
- Raw model without instruction training
- For text completion or custom fine-tuning
- **Don't use unless you know what you're doing**

### code
**Example:** `deepseek-coder:6.7b-instruct`
- Specialized for programming tasks
- Better at code generation and debugging

## Quantization Levels

Quantization compresses models by using lower precision. **This is critical for resource-constrained deployments.**

| Quantization | Size | Quality | RAM (for 8b model) |
|--------------|------|---------|-------------------|
| **q2_K** | ~25% | Noticeable loss | ~2 GB |
| **q3_K_M** | ~40% | Some loss | ~3 GB |
| **q4_0** | ~50% | Minimal loss | ~4 GB |
| **q4_K_M** | ~50% | Minimal loss (better) | ~4 GB |
| **q5_K_M** | ~60% | Nearly imperceptible | ~5 GB |
| **q6_K** | ~75% | Negligible loss | ~6 GB |
| **q8_0** | ~90% | Almost no loss | ~8 GB |
| **fp16** | 100% | No loss | ~16 GB |

**For resource-constrained setups: Use q4_0 or q4_K_M**

### Quick Memory Guide

**3B models:**
- q4_0: ~2 GB RAM

**7-8B models:**
- q4_0: ~4 GB RAM
- q5_K_M: ~5 GB RAM

**13-14B models:**
- q4_0: ~8 GB RAM

**70B models:**
- q4_0: ~40 GB RAM

## Quick Recommendations

**< 4 GB RAM:**
```bash
ollama pull phi:2.7b-q4_0
```

**4-8 GB RAM (recommended starting point):**
```bash
ollama pull llama3:8b-instruct-q4_0
```

**8-16 GB RAM:**
```bash
ollama pull llama3:8b-instruct-q5_K_M
```

## Common Questions

**Q: q4_0 vs q4_K_M?**  
A: q4_K_M is slightly better quality for almost the same size. Start with q4_0, try q4_K_M if you want better quality.

**Q: Instruct or base?**  
A: Almost always use instruct or chat. Base models are for advanced users only.

**Q: How do I estimate RAM needs?**  
A: For q4_0 models: ~0.5-0.6 GB per billion parameters. So an 8b-q4_0 needs ~4 GB RAM.
