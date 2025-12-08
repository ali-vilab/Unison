import os
import time
import gc
import copy
import random
import math
from tqdm import tqdm
from contextlib import contextmanager

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF

from wan.text2video import FlowUniPCMultistepScheduler
from vace import annotators
from vace.annotators.utils import read_image, read_mask, read_video_frames, save_one_video, save_one_image
from vace.configs import VACE_PREPROCCESS_CONFIGS
from vace.models.wan.wan_vace import WanVace


class ProjMLP(nn.Module):
    def __init__(self, input_dim, t5_dim, hidden_dim=2048, out_dim=None, dropout=0.1, target_seq_len=56, final_output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.t5_dim = t5_dim
        self.out_dim = out_dim or t5_dim
        self.target_seq_len = target_seq_len
        self.final_output_dim = final_output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim)
        )
        self.seq_adjust = nn.Linear(self.target_seq_len, final_output_dim)
        self.ln_in = nn.LayerNorm(input_dim)
        self.ln_out = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.ln_in(x)
        x = self.net(x) 
        x = self.ln_out(x)
        if seq_len < self.target_seq_len:
            padding = torch.zeros(batch_size, self.target_seq_len - seq_len, self.out_dim, 
                                device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        elif seq_len > self.target_seq_len:
            x_padded = x[:, :self.target_seq_len, :]
        else:
            x_padded = x
        x_transposed = x_padded.transpose(1, 2)
        x_adjusted = self.seq_adjust(x_transposed)
        output = x_adjusted.transpose(1, 2)
        return output


class WanVaceProj(WanVace):
    def generate(self,
                 input_prompt,
                 input_state,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        for i in range(len(context)):
            context[i] = torch.cat([context[i], input_state[i]], dim=0)

        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,**arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            self.vid_proc.set_seq_len(area // 12)

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images


def process_vace_data(
    task,
    video=None,
    image=None,
    mode=None,
    mask=None,
    bbox=None,
    label=None,
    caption=None,
    direction=None,
    expand_ratio=None,
    expand_num=None,
    maskaug_mode=None,
    maskaug_ratio=None,
    pre_save_dir=None,
    save_fps=16
):
    assert task in VACE_PREPROCCESS_CONFIGS, f"Unsupport task: [{task}]"
    assert video is not None or image is not None or bbox is not None, "Please specify the video or image or bbox."

    if bbox is not None:
        bbox = parse_bboxes(bbox)

    task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)[task]
    class_name = task_cfg.pop("NAME")
    input_params = task_cfg.pop("INPUTS")
    output_params = task_cfg.pop("OUTPUTS")

    fps = None
    input_data = copy.deepcopy(input_params)
    
    if 'video' in input_params or 'frames' in input_params:
        assert video is not None, "Please set video or check configs"
        frames, fps, width, height, num_frames = read_video_frames(video.split(",")[0], use_type='cv2', info=True)
        assert frames is not None, "Video read error"
        if 'video' in input_params:
            input_data['video'] = video
        if 'frames' in input_params:
            input_data['frames'] = frames
    
    if 'frames_2' in input_params and video is not None and len(video.split(",")) >= 2:
        frames, fps, width, height, num_frames = read_video_frames(video.split(",")[1], use_type='cv2', info=True)
        assert frames is not None, "Video read error"
        input_data['frames_2'] = frames
    
    if 'image' in input_params:
        assert image is not None, "Please set image or check configs"
        img, width, height = read_image(image.split(",")[0], use_type='pil', info=True)
        assert img is not None, "Image read error"
        input_data['image'] = img
    
    if 'image_2' in input_params and image is not None and len(image.split(",")) >= 2:
        img, width, height = read_image(image.split(",")[1], use_type='pil', info=True)
        assert img is not None, "Image read error"
        input_data['image_2'] = img
    
    if 'images' in input_params:
        assert image is not None, "Please set image or check configs"
        images = [read_image(path, use_type='pil', info=True)[0] for path in image.split(",")]
        input_data['images'] = images
    
    if 'mask' in input_params and mask is not None:
        mask_img, width, height = read_mask(mask.split(",")[0], use_type='pil', info=True)
        assert mask_img is not None, "Mask read error"
        input_data['mask'] = mask_img
    
    if 'bbox' in input_params and bbox is not None:
        input_data['bbox'] = bbox[0] if len(bbox) == 1 else bbox
    
    if 'label' in input_params and label is not None:
        input_data['label'] = label.split(',')
    
    if 'caption' in input_params:
        input_data['caption'] = caption
    
    if 'mode' in input_params:
        input_data['mode'] = mode
    if 'direction' in input_params and direction is not None:
        input_data['direction'] = direction.split(',')
    if 'expand_ratio' in input_params and expand_ratio is not None:
        input_data['expand_ratio'] = expand_ratio
    if 'expand_num' in input_params and expand_num is not None:
        input_data['expand_num'] = expand_num
    if 'mask_cfg' in input_params and maskaug_mode is not None:
        mask_cfg = {"mode": maskaug_mode}
        if maskaug_ratio is not None:
            mask_cfg["kwargs"] = {'expand_ratio': maskaug_ratio, 'expand_iters': 5}
        input_data['mask_cfg'] = mask_cfg

    pre_ins = getattr(annotators, class_name)(cfg=task_cfg, device=f'cuda:{os.getenv("RANK", 0)}')
    results = pre_ins.forward(**input_data)

    save_fps = fps if fps is not None else save_fps
    if pre_save_dir is None:
        pre_save_dir = os.path.join('processed', task, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    
    if not os.path.exists(pre_save_dir):
        os.makedirs(pre_save_dir)

    ret_data = {}
    
    if 'frames' in output_params:
        frames = results['frames'] if isinstance(results, dict) else results
        if frames is not None:
            save_path = os.path.join(pre_save_dir, f'src_video-{task}.mp4')
            save_one_video(save_path, frames, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data['src_video'] = save_path
    
    if 'masks' in output_params:
        frames = results['masks'] if isinstance(results, dict) else results
        if frames is not None:
            save_path = os.path.join(pre_save_dir, f'src_mask-{task}.mp4')
            save_one_video(save_path, frames, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data['src_mask'] = save_path
    
    if 'image' in output_params:
        ret_image = results['image'] if isinstance(results, dict) else results
        if ret_image is not None:
            save_path = os.path.join(pre_save_dir, f'src_ref_image-{task}.png')
            save_one_image(save_path, ret_image, use_type='pil')
            print(f"Save image result to {save_path}")
            ret_data['src_ref_images'] = save_path
    
    if 'images' in output_params:
        ret_images = results['images'] if isinstance(results, dict) else results
        if ret_images is not None:
            src_ref_images = []
            for i, img in enumerate(ret_images):
                if img is not None:
                    save_path = os.path.join(pre_save_dir, f'src_ref_image_{i}-{task}.png')
                    save_one_image(save_path, img, use_type='pil')
                    print(f"Save image result to {save_path}")
                    src_ref_images.append(save_path)
            if len(src_ref_images) > 0:
                ret_data['src_ref_images'] = ','.join(src_ref_images)
    
    if 'mask' in output_params:
        ret_image = results['mask'] if isinstance(results, dict) else results
        if ret_image is not None:
            save_path = os.path.join(pre_save_dir, f'src_mask-{task}.png')
            save_one_image(save_path, ret_image, use_type='pil')
            print(f"Save mask result to {save_path}")
            ret_data['src_mask_image'] = save_path
    
    return ret_data