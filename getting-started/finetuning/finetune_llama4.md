## Fine-Tuning Tutorial for Llama4 Models with torchtune

This tutorial shows how to perform Low-Rank Adaptation (LoRA) fine-tuning on Llama4 models using torchtune, based on the recent PR adding LoRA support for Llama4.

### Prerequisites

1. We need to use torchtune to perform LoRA fine-tuning. Now llama4 LORA fine-tune requires nightly build:
```bash
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
```

2. We also need Hugging Face access token (HF_TOKEN) for model download, please follow the instructions [here](https://huggingface.co/docs/hub/security-tokens) to get your own token.

### Tutorial
1. Download Llama4 Weights
Replace <HF_TOKEN> with your Hugging Face token:

```bash
tune download meta-llama/Llama-4-Scout-17B-16E-Instruct --output-dir /tmp/Llama-4-Scout-17B-16E-Instruct --hf-token $HF_TOKEN
```

Alternatively, you can use `huggingface-cli` to login then download the model weights.
```bash
huggingface-cli login --token $HF_TOKEN
tune download meta-llama/Llama-4-Scout-17B-16E-Instruct --output-dir /tmp/Llama-4-Scout-17B-16E-Instruct
```
This retrieves the model weights, tokenizer from Hugging Face.

2. Run LoRA Fine-Tuning for Llama4
To run LoRA fine-tuning, use the following command:
```bash
tune run --nproc_per_node 8 lora_finetune_distributed --config llama4/scout_17B_16E_lora
```
You can add specific overrides through the command line. For example, to use a larger batch_size:
```bash
  tune run --nproc_per_node 8 lora_finetune_distributed --config llama4/scout_17B_16E_lora batch_size=4 dataset.packed=True tokenizer.max_seq_len=2048
```
This will run LoRA fine-tuning on Llama4 model with 8 GPUs. It will requires around 400GB gpu memory to do Scout lora fine-tuning.

The config llama4/scout_17B_16E_lora is a config file that specifies the model, tokenizer, and training parameters. The dataset.packed=True and tokenizer.max_seq_len=2048 are additional arguments that specify the dataset and tokenizer settings.To learn more about the available options, please refer to the [YAML config documentation](https://pytorch.org/torchtune/stable/deep_dives/configs.html#config-tutorial-label)

With this setup, you can efficiently train LoRA adapters on Llama4 models using torchtune’s native recipes.

3. Full Parameter Fine-Tuning for Llama4 (Optional)
To run full parameter fine-tuning, use the following command:
```bash
  tune run --nproc_per_node 4  --nproc_per_node 8 full_finetune_distributed --config llama4/scout_17B_16E_full batch_size=4 dataset.packed=True tokenizer.max_seq_len=2048
  ```
This will run full parameter fine-tuning on Llama4 model with 8 GPUs. It will requires around 2200GB gpu memory to do Scout full parameter fine-tuning, which is about 4 8xH100 nodes.
