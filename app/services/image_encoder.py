from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import models, transforms
from loguru import logger


class ImageEncoder:
    """
    Extracts image embeddings
    using pretrained ResNet18.
    """

    def __init__(self):

        logger.info("Loading ResNet18 model")

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.model = torch.nn.Sequential(*list(model.children())[:-1])

        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def encode_image_url(self, image_url: str) -> np.ndarray | None:

        try:

            response = requests.get(image_url, timeout=5)

            image = Image.open(BytesIO(response.content)).convert("RGB")

            tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():

                embedding = self.model(tensor).squeeze().numpy()

            return embedding

        except Exception:

            return None
