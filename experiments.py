import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import open_clip
import torch
from PIL import Image

import cloob.clip as clip
from cloob.clip import _transform
from cloob.model import CLIPGeneral


class CLIP(object):
    def __init__(self, prompts, device):
        self.device = device

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu',
            pretrained='laion400m_e32'
        )
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
        clip_model.to(self.device)

        text_features = clip_model.encode_text(clip_tokenizer(prompts).to(self.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        self.model = clip_model
        self.preprocess = clip_preprocess
        self.text_features = text_features

    def __call__(self, frame):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image = self.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            image_features = self.model.encode_image(image.unsqueeze(0).to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ self.text_features.T)
            return sim


class CLOOB(object):
    def __init__(
            self, prompts, device,
            checkpoint_path="./checkpoints/cloob_rn50_yfcc_epoch_28.pt",
            configs_path="./cloob/model_configs/",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.configs_path = configs_path

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_config_file = os.path.join(self.configs_path, checkpoint['model_config_file'])

        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIPGeneral(**model_info)
        preprocess = _transform(model.visual.input_resolution, is_train=False)
        model.to(self.device)

        sd = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        if 'logit_scale_hopfield' in sd:
            sd.pop('logit_scale_hopfield', None)
        model.load_state_dict(sd)
        model.eval()

        with torch.no_grad():
            text_features = []
            for prompt in prompts:
                class_embeddings = model.encode_text(clip.tokenize([prompt]).to(self.device))
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=1).to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.text_features = text_features

    def __call__(self, frame):
        with torch.no_grad():
            image = self.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).to(device)
            image_embedding = self.model.encode_image(image.unsqueeze(0))
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            sim = (image_embedding @ self.text_features)
            return sim


if __name__ == "__main__":
    model_name = "CLIP"  # or "CLOOB"
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

    if model_name == "CLOOB":
        model = CLOOB(prompts, device)
    else:
        model = CLIP(prompts, device)

    sims = []

    for i, video_name in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(videos_path, video_name))
        assert cap.isOpened()
        sims.append([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            sim = model(frame)
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
