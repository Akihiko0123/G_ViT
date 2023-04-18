import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte

np.random.seed(0)
# 下記DataSetWrapperクラスのget_data_loadersメソッドから呼び出されている。
class Dataset():
    def __init__(self, csv_file, transform=None):
        # self.files_list = all_patches.csvに記載のファイル(デフォルトでは記載されていない)
        self.files_list = pd.read_csv(csv_file)
        # チェック用3.29
        print("ファイルリスト:",self.files_list.iloc[0,0])
#        sample_1 = Image.open(self.files_list.iloc[0,0])
#        sample_1
        # self.transform = 設定した拡張内容
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        # temp_path = idx行目に記載されたファイルパス
        temp_path = self.files_list.iloc[idx, 0]
        # img = temp_pathから画像パッチ(タイル)を開く
        img = Image.open(temp_path)
        # テンソルに変換する
        img = transforms.functional.to_tensor(img)
        # transformを行う場合は、実施する
        if self.transform:
#            print("拡張セット")
            sample = self.transform(img)
        return sample

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

# run.pyから呼び出され、dataとしてインスタンスを作成する
class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        # バッチサイズ
        self.batch_size = batch_size
        # configファイルのデータセット部分を読み込んでいる
        # 処理するプロセス(？)の数
        self.num_workers = num_workers
        # configファイルのデータセット部分を読み込んでいる
        # サイズ
        self.valid_size = valid_size
        # configファイルのデータセット部分を読み込んでいる
        # データ拡張時に色をランダムに変化させる範囲を計算するときに用いる係数
        self.s = s
        # configファイルのデータセット部分を読み込んでいる
        # 画像のサイズ(224 x 224 x 3 等)
        self.input_shape = eval(input_shape)

    # simclr.pyのSimCLRクラスのtrainメソッドより呼び出される
    def get_data_loaders(self):
        # data_augment = 拡張内容(色変化、ぼかしなど)
        data_augment = self._get_simclr_pipeline_transform()
        # train_dataset = Datasetクラスのインスタンス(csvはall_patches.csv、拡張は上記data_augmentを使用)
        train_dataset = Dataset(csv_file='all_patches.csv', transform=SimCLRDataTransform(data_augment))
#        # チェック用3.29
#        print(f"データ数：{train_dataset.__len__()}")
#        sample_x = train_dataset.__getitem__(0)
#        print("sample_x:",sample_x)
        # train_datasetを引数として学習用train_loaderと検証用valid_loaderをget_train_validation_data_loadersメソッドから取得する
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        # SimCLRクラス、trainメソッドの返り値とする。
        return train_loader, valid_loader
    # get_data_loadersメソッドより呼び出されるメソッドで、データ拡張を行う
    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        # 指定した範囲でランダムにコントラスト、彩度、色相を変化させる
        # (self.s=1なら、変動幅が明るさ・コントラスト、彩度0.8以上、色相0.2以上)
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        # data_transforms = 複数の拡張の組み合わせの実行結果(水平フリップ、80%の確率で色を変化、20%でグレースケール化、ぼかす)
        data_transforms = transforms.Compose([ToPIL(),
                                            #   transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms
    # get_data_loadersメソッドより呼び出される。学習データセットより作成
    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        # num_train = 学習データの数(all_patchesにリンクを記載したタイルの枚数)
        num_train = len(train_dataset)
        # チェック用(23'3.27)
        print("データイメージの数:",num_train)
        # インデックスindices = 学習データ数から作成したインデックスリスト(e.g. 7個の場合、[0,1,2,3,4,5,6])
        indices = list(range(num_train))
        # インデックスのリストをランダムに並べ替える(例：indices = [6,3,0,2,5,4,1])
        np.random.shuffle(indices)
        # split = (config.yaml内に設定した引数のvalid_size * データ数)の小数点以下を切り捨てて、整数に変換したもの
        #         つまり、splitは検証データの数。valid_sizeは学習データに対する検証データの割合を表す
        split = int(np.floor(self.valid_size * num_train))
        # データ数だけ項目を持つインデックスのリストindicesの前半split個がvalid_idx(検証インデックス), split以降の後半がtrain_idx(学習インデックス)
        train_idx, valid_idx = indices[split:], indices[:split]
        # チェック用
        print(f"学習インデックス:{train_idx},検証インデックス:{valid_idx}")

        # define samplers for obtaining training and validation batches
        # train_sampler = 学習データのインデックスを基にシャッフルして、DataLoaderメソッドを通して学習データだけの
        # データローダーを作成するもの
        # valid_sampler = 同じくDataloaderメソッドで指定すると検証データだけのデータローダーを作成してくれるもの
        # SubsetRandomSamplerの使い方(https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a)
        # Dataloaderクラスでミニバッチを作成する際に、元のデータは同じ物(train_dataset)を指定していても、
        # samlerオプションとしてそれぞれのサンプラーを指定することで、ランダムかつ均等に学習データと検証データを取得できるようになる。
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # サンプラーを使って、学習用と検証用のデータローダーを作成している
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        #tmp = train_loader.__iter__()
        #xx = next(tmp)
        #print("xx[0]:",xx[0])
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        # 学習ローダーと検証ローダーを返す
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
