# llm-toys

Production-ready small LLMs fine-tuned for diverse tasks, all without an OpenAI key.


## Training

### Paraphrase + Tone change:
```bash
python llm_toys/train.py \
    --task_type paraphrase_tone \
    --model_name tiiuae/falcon-7b \
    --max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --use_aim
```