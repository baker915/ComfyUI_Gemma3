import os
import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image
import folder_paths
from pathlib import Path
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


class Gemma3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        gemma_dir = os.path.join(folder_paths.models_dir, "Gemma3")
        gguf_models = []

        if os.path.exists(gemma_dir):
            for file in os.listdir(gemma_dir):
                if file.endswith('.gguf'):
                    gguf_models.append(file)

        if not gguf_models:
            gguf_models = ["no_gguf_models_found"]

        return {
            "required": {
                "model_id": (
                ["google/gemma-3-27b-it", "google/gemma-3-12b-it", "google/gemma-3-1b-it", "google/gemma-3-4b-it"],
                {"default": "google/gemma-3-27b-it"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
                "use_gguf": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "local_gemma3_model_path": (gguf_models, {"default": gguf_models[0]}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "gemma3"

    def load_model(self, model_id, load_local_model, use_gguf, *args, **kwargs):
        if use_gguf:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("需要安装 llama-cpp-python: pip install llama-cpp-python")

            gemma_dir = os.path.join(folder_paths.models_dir, "Gemma3")
            model_filename = kwargs.get("local_gemma3_model_path", "gemma-3-27b-it-Q4_K_M.gguf")
            model_path = os.path.join(gemma_dir, model_filename)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GGUF 模型文件不存在: {model_path}")

            n_gpu_layers = kwargs.get("n_gpu_layers", -1)

            model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=128000,
                verbose=False
            )
            return (model, None)

        device = mm.get_torch_device()
        if load_local_model:
            model_id = kwargs.get("local_gemma3_model_path", model_id)
        else:
            gemma_dir = os.path.join(folder_paths.models_dir, "Gemma3")
            os.makedirs(gemma_dir, exist_ok=True)

            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, cache_dir=gemma_dir, device_map="auto"
            ).eval().to(device)
            processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=gemma_dir
            )
            return (model, processor)

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        return (model, processor)


class ApplyGemma3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "processor": ("PROCESSOR",),
                "max_new_tokens": ("INT", {"default": 100, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "do_sample": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_gemma3"
    CATEGORY = "gemma3"

    def apply_gemma3(self, model, processor, max_new_tokens, temperature, top_p, do_sample, system_prompt=None, user_prompt=None, image=None):
        if processor is None:
            return self._apply_gguf(model, system_prompt, user_prompt, max_new_tokens, temperature, top_p, do_sample, image)
        else:
            return self._apply_transformers(model, processor, system_prompt, user_prompt, max_new_tokens, temperature, top_p, do_sample, image)

    def _apply_transformers(self, model, processor, system_prompt, user_prompt, max_new_tokens, temperature, top_p, do_sample, image=None):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
        ]

        if image is not None:
            image_pil = tensor2pil(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": user_prompt}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            })

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return (decoded,)

    def _apply_gguf(self, model, system_prompt, user_prompt, max_new_tokens, temperature, top_p, do_sample, image=None):
        if image is not None:
            raise NotImplementedError("GGUF 模型暂不支持图像输入，请使用纯文本推理")

        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|model|>\n"

        output = model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )

        return (output['choices'][0]['text'],)


NODE_CLASS_MAPPINGS = {
    "Gemma3ModelLoader": Gemma3ModelLoader,
    "ApplyGemma3": ApplyGemma3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma3ModelLoader": "Gemma3 Model Loader",
    "ApplyGemma3": "Apply Gemma3",
}
