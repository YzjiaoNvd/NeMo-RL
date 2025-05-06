# Evaluation

This document explains how to use an evaluation script for assessing model capabilities.

## Start Evaluation

To run the evaluation, you can use the default configuration file or specify a custom one.

### Start Script

**Evaluate Standard Models:**

To run evaluation using a model directly from Hugging Face Hub or a local path already in HF format, use the `run_eval.py` script.

```sh
# To run the evaluation with default config (examples/configs/eval.yaml)
uv run python examples/run_eval.py

# Specify a custom config file
uv run python examples/run_eval.py --config path/to/custom_config.yaml

# Override specific config values via command line (e.g., model name)
uv run python examples/run_eval.py generation.model_name="Qwen/Qwen2.5-Math-7B-Instruct"
```

**Evaluate Models Trained with DCP Checkpoints (GRPO/SFT):**

If you have trained a model using GRPO or SFT and saved the checkpoint in the Pytorch DCP format, you first need to convert it to the Hugging Face format before running evaluation.

1.  **Convert DCP to HF:**
    Use the `examples/convert_dcp_to_hf.py` script. You'll need the path to the training configuration file (`config.yaml`), the DCP checkpoint directory, and specify an output path for the HF format model.

    ```sh
    # Example for a GRPO checkpoint at step 170
    uv run python examples/convert_dcp_to_hf.py \
        --config results/grpo/step_170/config.yaml \
        --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
        --hf-ckpt-path results/grpo/hf
    ```
    *Note: Adjust the paths according to your training output directory structure.*

2.  **Run Evaluation on Converted Model:**
    Once the conversion is complete, run the evaluation script, overriding the `generation.model_name` to point to the directory containing the converted HF model.

    ```sh
    # Example using the converted HF model from the previous step
    uv run python examples/run_eval.py generation.model_name=$PWD/results/grpo/hf
    ```

### Example Output

```
============================================================
model_name='Qwen2.5-Math-1.5B-Instruct' dataset_name='aime_2024'
score=0.10 (3.0/30)
============================================================
```

## Example Configuration File

You can find an example evaluation configuration file [here](../../examples/configs/eval.yaml).

### Prompt Template Configuration

Always remember to use the same `prompt_file` and `system_prompt_file` that were used during training.

For open-source models, we recommend setting `prompt_file=null` and `system_prompt_file=null` to allow them to use their native chat templates.
