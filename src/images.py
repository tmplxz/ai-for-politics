import os
import subprocess


def generate_images_uno(visual_details, image_gen_path, image_gen_call, save_path, base_img, n_images=2, guidance=10, num_steps=25):
    subdir = os.path.join(save_path, f'img_{os.path.basename(base_img).split(".")[0]}_guid{guidance}_nsteps{num_steps}')
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    image_gen_call = image_gen_call.format(prompt=f"Add {visual_details}", base=base_img, dir=subdir, n_img=n_images, guid=guidance, nsteps=num_steps)
    print(image_gen_call)
    exitcode = subprocess.run(image_gen_call, shell=True, cwd=image_gen_path)
    if exitcode.returncode != 0:
        print("Error in generating image")
        return False
    return True


def load_model_diffusers(model_name):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_name, device_map="balanced")
    return pipe


def generate_images_diffusers(model, visual_details, save_path, guidance=0., num_steps=4, n_images=4):
    import torch
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    prompt = "City in Germany, with additional " + visual_details
    images = model(
        prompt,
        height=512,
        width=512,
        max_sequence_length=256,
        generator=[torch.Generator("cuda").manual_seed(i) for i in range(n_images)],  # 4 images, different seeds
        num_images_per_prompt=n_images,
        num_inference_steps=num_steps,
        guidance_scale=guidance
    )
    for idx, image in enumerate(images.images):
        image.save(os.path.join(save_path, f"0_{idx}.png"))
    return True