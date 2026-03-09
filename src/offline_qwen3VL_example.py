"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
# -*- coding: utf-8 -*-
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams


#os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

# Qwen3-VL-MOE
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=False, # all empty list or None when no video
        return_video_metadata=False
    )
    print(f"text after chat template: {text}\nimage_patch_size: {processor.image_processor.patch_size}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data
    }


if __name__ == '__main__':
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    message_text_prompt = f"""
    Task: Generate 3 technical Q&A pairs. 
    IMPORTANT: At least two of the answers MUST explicitly refer to the provided image. 
    Use phrases like "As seen in the image," "According to the diagram," or "Looking at the hardware labels."
    
    If the image shows specific component labels (like 'Address switch', 'CAN bus', or 'Terminating resistor'), refer to them by name in your answers.

    Format: Return ONLY a JSON list of objects with 'question' and 'answer' keys.
    """
    image_paths = ["data/Chapter17_image_1.png", "data/Chapter17_image_1.png"]
    messages = [
        {
            "role": "user",
            "content": [
              {
                  "type": "image",
                  "image": img_path,
              },
              {"type": "text", "text": message_text_prompt},
            ],
        } for img_path in image_paths
    ]

    # TODO: change to your own checkpoint path
    checkpoint_path = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]
    engine_args = EngineArgs(
        model=checkpoint_path,
        max_model_len=-1,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": [],
        },
        limit_mm_per_prompt={'image': 1, 'video': 0, 'audio': 0, 'vision_chunk': 0}, # 0 out not needed modality
        seed=0,
        trust_remote_code=True,
        gpu_memory_utilization=0.70,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count()
    )

    engine_args = asdict(engine_args)

    llm = LLM(
        **engine_args
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated response: {generated_text!r}")
