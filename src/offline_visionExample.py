import PIL
from vllm import LLM, SamplingParams
from pathlib import Path

data = PIL.Image.open("data/example.jpg")
model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
model_path = Path("/ds/models/llms/Qwen/Qwen3-VL-30B-A3B-Instruct")

if model_path.exists():
    print(f"Loading from {model_path}")
    llm = LLM(
        model=str(model_path),
        max_num_seqs=5,
    )
else:
    print(f"Need to download model {model_name} to cache")
    llm = LLM(
        model=model_name,
        max_num_seqs=5
    )
stop_token_ids = None

question = "Describe the image"
prompt_template = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                      f"{question}<|im_end|>\n"
                      "<|im_start|>assistant\n")

inputs = {
                "prompt": prompt_template,
                "multi_modal_data": {
                    "image": data
                },
            }

sampling_params = SamplingParams(temperature=0.2,
                                          max_tokens=128,
                                          stop_token_ids=stop_token_ids,
                                          )

outputs = llm.generate(inputs, sampling_params=sampling_params)
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

print("manual shutdown...", flush=True)
llm.llm_engine.engine_core.shutdown() # fixing EngineCore died unexpectedly when not in Main()
