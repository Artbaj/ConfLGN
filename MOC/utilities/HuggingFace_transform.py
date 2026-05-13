from datasets import load_dataset as load_hf_dataset
from torch.utils.data import Dataset
from torchvision import transforms

class PatchCamelyonTorchDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        image = item["image"]
        label = int(item["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label