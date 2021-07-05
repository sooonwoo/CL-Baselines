import os 
from PIL import Image
import torchvision.datasets as datasets

class COCODataset(datasets.CocoDetection):
    def __init__(self, root, annFile, transform):
        super().__init__(root, annFile, transform)
        print(root)
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, -1

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __len__(self) -> int:
        return len(self.ids)