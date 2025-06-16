from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            print(f"Length before: {len(self.images)}, {len(self.targets)}")
            mask = self.targets != remove_class
            self.images = self.images[mask]
            self.targets = self.targets[mask]
            print(f"Length after: {len(self.images)}, {len(self.targets)}")

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        sidro = self.targets[index].item()
        
        # Dohvati sve razrede koji nisu isti kao razred sidra
        negclass = [cls for cls in self.classes if cls != sidro]
        
        # Slučajno odaberi jedan od negativnih razreda
        negative_class = choice(negclass)
        
        # Slučajno odaberi sliku iz odabranog negativnog razreda
        neg_index = choice(self.target2indices[negative_class])
        
        return neg_index


    def _sample_positive(self, index):
        sidro = self.targets[index].item()
        
        posclass = [cls for cls in self.classes if cls == sidro]

        positive = choice(posclass)

        pos_indx = choice(self.target2indices[positive])

        while pos_indx == index:
            pos_indx = choice(self.target2indices[positive])
        
        return pos_indx


    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = MNISTMetricDataset()
    k= dataset._sample_negative(4)
    j= dataset._sample_positive(5)
    print(k)
    print("#"*30)
    print(j)