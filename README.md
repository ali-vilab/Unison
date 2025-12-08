# Unison: A Fully Automatic, Task-Universal, and Low-Cost Framework for Unified Understanding and Generation

### [Project Page](https://ali-vilab.github.io/Unison-Page) | [Paper (ArXiv)](xxxx)

## Introduction
Unison is a two-stage framework for unified understanding and generation tasks. Trained at minimal cost with only 500K samples and 50 GPU hours, Unison supports a wide range of understanding tasks across text, image, and video, as well as generation tasks including text-to-visual generation, editing, controllable generation, and IP-based reference generation, totaling 12 types of tasks. Notably, Unison can automatically parse user intention, identify task types, and extract necessary meta-information, enabling full automation of multimodal workflows without human intervention.

<img width="800" alt="image" src="./assets/framework_design_and_training.png">


## ⚙ : Setup

You can setup for Unison inference by running:
```bash
    pip install -r requirements.txt
    pip install flash_attn==2.7.4.post1 --no-build-isolation
    pip install wan@git+https://github.com/Wan-Video/Wan2.1
    pip install vace@git+https://github.com/ali-vilab/VACE
```
You should download the Annotators by running:
```bash
    mkdir models
    huggingface-cli download --resume-download ali-vilab/VACE-Annotators --local-dir models/VACE-Annotators
    pip install models/VACE-Annotators/gdino/groundingdino-0.1.0-cp310-cp310-linux_x86_64.whl
    pip install models/VACE-Annotators/sam2/SAM_2-1.0-cp310-cp310-linux_x86_64.whl
```

## 💻 : Test

You should first download the pre-trained models: the stage-one model [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and the stage-two model [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B). You also need to download the LoRA we trained to equip the stage-one model with planning capability, as well as the projector trained to align the two stages of models. Here is the [link](xxxx)。

Next, you need to modify the `model_name_or_path`, `adapter_name_or_path` and `export_dir` in the `merge_lora.yaml` file to your own paths, and run the following command to merge the stage-one model and the LoRA:

```bash
    llamafactory-cli export merge_lora.yaml
```

Then, change `tie_word_embeddings` in the merged model's `config.json` to false. 

You need to update `qwenvl_path` in `run.sh` to the path of the merged stage-one model (with LoRA), set `vace_path` to the path of the stage-two model, and `proj_path` to the path of the projector. After that, you can run the following command for inference:

```bash
    bash run.sh
```

It is worth noting that the PROMPT_TEXT should be in string format, which includes the text input and the corresponding paths to visual content (such as images, videos, and masks), marked with `###PATH###`. Examples for different tasks are provided in `run.sh`, which you can run to try out.


## 🎉 : Acknowledgments

This work is based on [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL), [VACE](https://github.com/ali-vilab/VACE), [Wan2.1](https://github.com/Wan-Video/Wan2.1). We extend our gratitude to the contributors of these projects!


## 📖 : Citation

```bibtex
@article{zhao2025unison,
  title={Unison: A Fully Automatic, Task-Universal, and Low-Cost Framework for Unified Understanding and Generation},
  author={Shihao Zhao, Yitong Chen, Zeyinzi Jiang, Bojia Zi, Shaozhe Hao, Yu Liu, Chaojie Mao, Kwan-Yee~K.},
  journal={xxxx},
  year={2025}
}
```
