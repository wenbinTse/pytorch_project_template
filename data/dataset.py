from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

voc_class_to_name = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

voc_name_to_class = {x: i for i, x in enumerate(voc_class_to_name)}

voc_train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

voc_val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

class VOCClassification(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        assert mode in ['train', 'test']
        self.args = args
        self.root = args.voc_root
        self.mode = mode
        self.transform = voc_train_transforms if mode == 'train' else voc_val_transforms
        with open(os.path.join(self.root, 'class.txt')) as f:
            records = f.readlines()
            records_size = len(records)
            self.records = records[:int(records_size * 0.8)] if mode == 'train' \
                else records[int(records_size * 0.8):]
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, i):
        image_id, class_id = self.records[i].split()
        class_id = int(class_id)
        image_name = os.path.join(self.root, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(image_name)
        image = self.transform(image)
        return image, class_id, image_name


if __name__ == '__main__':
    import options
    dataset = VOCClassification(options.args, voc_train_transforms)
    print(dataset.__getitem__(0), len(dataset))
    