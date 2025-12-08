import os
import re
import argparse
import ffmpeg
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import wan
from wan.utils.utils import cache_video, cache_image
from vace.models.wan.configs import WAN_CONFIGS

from module import ProjMLP, WanVaceProj, process_vace_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwenvl_path", type=str, required=True)
    parser.add_argument("--vace_path", type=str, required=True)
    parser.add_argument("--proj_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser


def main():
    args = get_parser().parse_args()
    cfg = WAN_CONFIGS["vace-1.3B"]

    prompt = args.prompt
    visual_content_list = re.findall(r'###(.*?)###', prompt)
    for visual_content_path in visual_content_list:
        if visual_content_path.endswith((".png", ".jpg", ".jpeg")):
            PAD_TOKEN = "<IMGPAD>"
        elif visual_content_path.endswith((".mp4", ".avi", ".mov")):
            PAD_TOKEN = "<VIDPAD>"
        else:
            assert False, "Unsupported file type"
        prompt = prompt.replace(f"###{visual_content_path}###", PAD_TOKEN)
    
    qwenvl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwenvl_path, device_map="auto").to("cuda")
    qwenvl_processor = AutoProcessor.from_pretrained(args.qwenvl_path)

    wan_vace = WanVaceProj(config=cfg, checkpoint_dir=args.vace_path, device_id=0)
    wan_vace.model.text_len = 2048
    projector = ProjMLP(input_dim=qwenvl_model.config.text_config.hidden_size, t5_dim=4096).to("cuda")
    state_dict = torch.load(args.proj_path)
    projector.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    with torch.no_grad():
        inputs = qwenvl_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to("cuda")
        generated_ids = qwenvl_model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = qwenvl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        output_text = output_text[0].replace('<|im_end|>', '').strip()

    size = (832, 480)
    src_mask = src_video = src_ref = None
    if '<CFI>' in output_text:
        print(output_text)
        num_frames = 1
        if '<BORES>' in output_text:
            match = re.search(r'<BORES>([^<]+)<EORES>', output_text)
            match_str = match.group(1)
            width, height = map(int, match_str.split(','))
            size = (width, height)
        if '<BOEDIT>' in output_text:
            match = re.search(r'<BOEDIT>([^<]+)<EOEDIT>', output_text)
            match_str = match.group(1)
            mask_id, source_id = map(int, match_str.split(','))
            mask_path = visual_content_list[mask_id]
            src_path = visual_content_list[source_id]
            result = process_vace_data(task="inpainting", mode="mask", video=src_path, mask=mask_path, save_fps=24)
            src_mask = result['src_mask']
            src_video = result['src_video']
        if '<REF>' in output_text:
            ref_path = visual_content_list[0]
            result = process_vace_data(task="image_reference", mode="plain", image=ref_path)
            src_ref = result['src_ref_images']
    elif '<CFV>' in output_text:
        print(output_text)
        num_frames = 81
        if '<BORES>' in output_text:
            match = re.search(r'<BORES>([^<]+)<EORES>', output_text)
            match_str = match.group(1)
            width, height = map(int, match_str.split(','))
            size = (width, height)
        if '<BONF>' in output_text:
            match = re.search(r'<BONF>([^<]+)<EONF>', output_text)
            match_str = match.group(1)
            num_frames = int(match_str) + 1
        if '<BOFIDX>' in output_text:
            match = re.search(r'<BOFIDX>([^<]+)<EOFIDX>', output_text)
            match_str = match.group(1)
            mode = "firstframe" if match_str == "match_str" else "lastframe"
            result = process_vace_data(task="frameref", mode=mode, image=visual_content_list[0])
            src_mask = result['src_mask']
            src_video = result['src_video']
        if '<BOEDIT>' in output_text:
            src_path = visual_content_list[0]
            result = process_vace_data(task="inpainting", mode="salient", video=src_path, save_fps=24)
            src_mask = result['src_mask']
            src_video = result['src_video']
        if '<REF>' in output_text:
            ref_path = visual_content_list[0]
            result = process_vace_data(task="image_reference", mode="plain", image=ref_path)
            src_ref = result['src_ref_images']
        if '<CTRL>' in output_text:
            src_video = visual_content_list[0]
    else:
        if len(visual_content_list) == 0:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": visual_content_list[0]}]}]
        with torch.no_grad():
            inputs = qwenvl_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = inputs.to("cuda")
            generated_ids = qwenvl_model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = qwenvl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            output_text = output_text[0].replace('<|im_end|>', '').strip()
            print(output_text)
        return

    with torch.no_grad():
        inputs_states = qwenvl_processor.apply_chat_template(
            messages, 
            tokenize=True,
            add_generation_prompt=False, 
            return_tensors="pt", 
            padding=True
        ).to("cuda")
        qwenvl_outputs = qwenvl_model.model(inputs_states).last_hidden_state.to("cuda")
        qwenvl_proj_feat = projector(qwenvl_outputs[:, 1:, :].float())

    src_video, src_mask, src_ref_images = wan_vace.prepare_source([src_video], [src_mask], [None if src_ref is None else src_ref.split(',')], num_frames, size, "cuda")
    video = wan_vace.generate(
        prompt,
        qwenvl_proj_feat,
        src_video,
        src_mask,
        src_ref_images,
        size=size,
        frame_num=num_frames,
        seed=0
    )

    os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
    if num_frames == 1:
        cache_image(tensor=video[:, 0, ...], save_file=args.save_path, nrow=1, normalize=True, value_range=(-1, 1))
    else:
        cache_video(tensor=video[None], save_file=args.save_path, fps=24, nrow=1, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()