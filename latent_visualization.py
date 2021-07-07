import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import random

class dataset(Dataset):

    def __init__(self,root_dir,n_sample, transform=True):

        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.image_path = []

        if not os.path.isdir(self.root_dir):
            raise NameError("{} is not a dir".format(root_dir))

        for f in os.listdir(self.root_dir):
            self.image_path.append(os.path.join(root_dir,f))
        self.len_path_list = len(self.image_path)
        random.shuffle(self.image_path)
        self.image_path = self.image_path[:min(self.len_path_list,n_sample)]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.image_path[idx % len(self.image_path)]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        sample = {'image': img}

        return sample

def get_feature(path, n_sample):
    data = dataset(path,n_sample)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64)
    model = models.resnet101(pretrained=True).to('cuda:0')
    model.eval()

    feature = []
    for batch in tqdm(dataloader):
        images = batch['image'].to('cuda:0')

        with torch.no_grad():
            output = model.forward(images)
        
        current_outputs = output.cpu().numpy()
        
        feature.append(current_outputs)

    return np.concatenate(feature)

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', action='store', type=str, nargs='*')
    parser.add_argument('--label', action='store', type=str, nargs='*')
    parser.add_argument('--n_components', default=3, type=int)
    parser.add_argument('--perplexity', default=30, type=int)
    parser.add_argument('--n_iter', default=4000, type=int)
    parser.add_argument('--n_sample', default=100, type=int)
    parser.add_argument('--path_save', default="", type=str)
    args = parser.parse_args()

    if len(args.path) != len(args.label):
        raise NameError("number of path must equal to number of label")

    feature_list = []
    for path in args.path:
        feature_list.append(get_feature(path, args.n_sample))

    label_list = []
    for i,label in enumerate(args.label):
        label_list.append(np.array([label] * feature_list[i].shape[0]))

    X = np.concatenate(feature_list)
    y = np.concatenate(label_list)
    print(X.shape)
    print(y.shape)
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    data_subset = df[feat_cols].values

    tsne = TSNE(n_components=args.n_components, verbose=1, perplexity=args.perplexity, n_iter=args.n_iter)
    tsne_results = tsne.fit_transform(data_subset)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,1], y=tsne_results[:,2],
        hue="y",
        data=df,
        palette="Set1",
        legend="full",
        alpha=0.3
    )
    plt.savefig(os.path.join(args.path_save,"t-SNE.png"))

if __name__ == '__main__':
    main()