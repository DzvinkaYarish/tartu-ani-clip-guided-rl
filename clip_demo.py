import gym
from stable_baselines3 import DQN
import cv2
import open_clip
from PIL import Image
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import math
import utils


if __name__ == "__main__":
    prompts = [
        # "a photo of a cat between two dogs",
        "a photo of a cat at the bottom between two dogs",
        "a photo of a cat between two photos of dogs at the bottom",
        "a photo of a cat between two photos of dogs at the bottom, horizontally aligned",
        # "a realistic photo of a student who is studying at the university"
    ]
    # Strategy can be "general" and "positive-negative". "positive-negative" means you have to provide only
    # two prompts (positive and negative) and the difference between them will be computed (positive - negative).
    # The last commented prompt above is an example of the negative prompt
    prompt_strategy = "general"
    videos_path = "/content/"
    videos = [
        "success.mp4",
        "partial_success.mp4",
        "fail1.mp4",
        "fail2.mp4",
        "fail3.mp4",
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
                                                                           pretrained='laion400m_e32')
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    clip_model.to(device)

    text_features = clip_model.encode_text(clip_tokenizer(prompts).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sims = []

    for i, video_name in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(videos_path, video_name))
        assert cap.isOpened()
        sims.append([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad(), torch.cuda.amp.autocast():
                image = clip_preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                image_features = clip_model.encode_image(image.unsqueeze(0).to(device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                sim = (image_features @ text_features.T)
            sim = sim.cpu().detach().numpy()[0]
            if prompt_strategy == "positive-negative":
                sims[i].append(sim[0] - sim[1])
            else:
                sims[i].append(sim)
        cap.release()
    cv2.destroyAllWindows()

    for i, prompt in enumerate(prompts):
        print(f"{i}: {prompt}")

    plt.rcParams["figure.figsize"] = (20, 20)
    grid_size = int(math.sqrt(len(videos))) + len(videos) % 2
    for i, video_name in enumerate(videos):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.plot(sims[i])
        plt.legend(list(range(len(prompts))))
        plt.title(video_name)
    plt.show()
