PROMPT_TEXT="There are 100 books in a library. On Monday, 40 books are borrowed. On Tuesday, 20 books are returned. How many books are now in the library?"
# PROMPT_TEXT="###./assets/image_understanding.jpg### What is the dark sauce in the bowl? Choices: A) tomato sauce B) bbq sauce C) salsa D) gravy Answer with the option's letter from the given choices directly."
# PROMPT_TEXT="Create an 480P image of a clownfish swims among the tentacles of a sea anemone, showcasing its vibrant orange and white stripes."
# PROMPT_TEXT="I have an image ###./assets/image_edit_source.png### and a mask ###./assets/image_edit_mask.png###. Apply the mask to the source image and alter the content to show them having tea party."
# PROMPT_TEXT="###./assets/image_ref.png### Please generate an ID photo for this woman, depicting her working in front of a computer in her room. The scene should include books scattered nearby, with a very warm layout featuring wooden walls and a warm-toned desk lamp."
# PROMPT_TEXT="Please create a video with a width of 768 pixels and a height of 480 pixels, that is 2 seconds long, of a lion walking around in its habitat."
# PROMPT_TEXT="Based on the provided last frame ###./assets/video_i2v.jpg###, generate a video that showcases a magnificent natural landscape. The camera remains stationary, capturing the rolling mountains and vast sky. The mountain ridges stand out distinctly against the blue sky, while clouds drift slowly through the valleys, creating a dynamic scene."
# PROMPT_TEXT="Based on the provided face ID ###./assets/video_ref.png###, generate a video featuring a close-up, eye-level shot of a foreign man. He is wearing a black outer coat over a black and red striped sweater, swaying his body and arms as if speaking. In the background, there is a black wall illuminated by purple lighting and a large screen. On the screen, a figure in yellow clothing is holding a yellow shield with a black pattern, alongside other figures in white, with the yellow ground visualized."
# PROMPT_TEXT="Here is the mask <VIDPAD>. Make the bird in this video ###./assets/video_edit_source.mp4### appear cleaner and neater, presenting it in white."
# PROMPT_TEXT="###./assets/video_ctrl.mp4### Following the provided pose information, please create a video showing a construction worker in workwear and a safety helmet diligently working at a construction site. He is holding a tablet, seemingly recording or reviewing information. In the background, buildings under construction and some construction materials are visible. The entire video is shot from a fixed angle, emphasizing the worker's focus and his working environment."

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --qwenvl_path "/Path/merge" \
    --vace_path "/Path/VACE-Wan2.1-1.3B-Preview" \
    --proj_path "/Path/projector.pth" \
    --prompt "$PROMPT_TEXT" \
    --save_path "/Path/result.png"
