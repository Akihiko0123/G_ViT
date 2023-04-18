import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

import torch.nn.functional as F
# LinuxかWindowsか判別用
import platform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]
    
def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

# main.pyより呼び出し,グラフデータのパス(=root)、train_set引数で指定したファイルの中身(=ids)が引数
class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, target_patch_size=-1):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample

        引数:

        ファイルディレクトリ(文字列)：　全ての入力画像のディレクトリ
        transform(呼び出し可能, オプション)：サンプルに適用されるオプションの拡張
        """
        # root = data_path+"\\"or"/"
        self.root = root
        self.ids = ids
        # チェック用
        print(f"[dataset.py]self.root:{self.root}")
        print(f"[dataset.py]self.ids:{self.ids}")

        info_t = self.ids[0].replace('\n', '')
        print(f"[dataset.py]info:{info_t}")
        file_name_t,label_t = info_t.split('\t')[0].rsplit('.', 1)[0], info_t.split('\t')[1]
        print(f"[dataset.py]file_name:{file_name_t}")
        print(f"[dataset.py]label:{label_t}")
        #self.target_patch_size = target_patch_size
        # デフォルトで3クラス(正常, 腺がん, 原発性肺扁平上皮癌)
        #self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}        #
        self.classdict = {'normal': 0, 'tumor': 1}        #
        #self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
        # モード：双線形(画像を回転、拡大、変形する際の隙間の補完法の１つ)
        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        sample = {}
        # idsはtrain_set引数で指定したファイル(デフォルト：train_set.txt)で、
        # ids[0]の場合、ファイルの1行目が文字列形式で取り出される。
        # \nは改行記号なので、これを削除することで始めて1行目を取り出すことができるが、
        # テキストファイルのスペースをTabで作っている場合、タブ記号('\t')はついたまま。
        info = self.ids[index].replace('\n', '')
        # train_set.txtがタブでスペースを取って記載していることが想定されているため,
        # タブ記号"\t"で分割すれば情報を分けて取り出せる
        # どうやら想定では、train_set.txtには
        # ptファイルが入ったフォルダ名？かファイル名？とタブ区切りのラベルが各行に記載されている。
        # rsplit('.',1)[0] => 後ろから1つ目の'.'だけを区切って
        # フォルダ名orファイル名側のパスだけを拾う(拡張子を取り除く)
        # file_name = ptファイルのフォルダパスorファイルパス(拡張子なし)
        # label = ラベル
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        # site = ファイルがあるディレクトリの上のディレクトリ名orファイルがあるディレクトリ名
        # file_name = ptファイルのディレクトリ名かファイル名
#        site, file_name = file_name.split('/')
            # Windows対応追加
        if platform.system() == "Linux":
            site, file_name = file_name.split('/')
        elif platform.system() == "Windows":
            print("[dataset.py]file_name.split:",file_name.split('\\'))
            site, file_name = file_name.split('\\')

        # siteのディレクトリ名がLUADやLSCCの場合はLUNGに変換して出力ファイルパスの一部に指定している。
        # if site =='CCRCC':
        #     file_path = self.root + 'CPTAC_CCRCC_features/simclr_files'
        if site =='LUAD' or site =='LSCC':
            site = 'LUNG'
        # ptファイルのディレクトリはdata_pathで指定したディレクトリ内(~\tmi2022-main\graphs?)にある？
        file_path = self.root + 'CPTAC_{}_features/simclr_files'.format(site)       #_pre# with # rushin

        # For NLST only
        # ディレクトリ名がNLSTの場合は出力先が独自のフォルダ名になる。
        if site =='NLST':
            file_path = self.root + 'NLST_Lung_features/simclr_files'

        # For TCGA only
        # ディレクトリ名がTCGAの場合は出力先が独自のフォルダ名になる。
        if site =='TCGA':
            file_name = info.split('\t')[0]
#            _, file_name = file_name.split('/')
            # Windows対応追加
            if platform.system() == "Linux":
                _, file_name = file_name.split('/')
            elif platform.system() == "Windows":
                _, file_name = file_name.split('\\')
            # グラフデータパスに'TCGA_LUNG_features/simclr_files'と言うパスをつなげたもの
            file_path = self.root + 'TCGA_LUNG_features/simclr_files'       #_resnet_with
        # classdictの例：self.classdict = {'normal': 0, 'tumor': 1}
        # sample['label'] = labelに合った数値(tumorなら1, normalなら0)
        sample['label'] = self.classdict[label]
        # チェック用
        print(f"[dataset.py]sample['label']:{self.classdict[label]}")
        # sample['id'] = スライドの特徴ファイル(全パッチから作成)か特徴ファイルが入ったフォルダの名称
        sample['id'] = file_name
        # チェック用
        print(f"[dataset.py]sample['id']:{file_name}")

        #feature_path = os.path.join(self.root, file_name, 'features.pt')
        # 特徴ファイルパスだが、
        # ~\\tmi2022-main\\graphs\\CPTAC_{site}_features\\simclr_files\\output\\features.ptになる
        #              root                                             file_name 
        #                          file_path                            file_name      
        # これらのフォルダ構成は事前に用意しておく必要がありそう
        feature_path = os.path.join(file_path, file_name, 'features.pt')

        '''
        モデルのパラメータとグラフ情報を同じ形で読み込んでいる
        '''
        if os.path.exists(feature_path):
            # map_location=lambda storage, loc: storageで、モデルの各層のパラメータを各層毎にロード
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, 512)

        #adj_s_path = os.path.join(self.root, file_name, 'adj_s.pt')
        adj_s_path = os.path.join(file_path, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            # map_location=lambda storage, loc: storageで、モデルの各層のパラメータを各層毎にロード
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        #features = features.unsqueeze(0)
        # sample['image'] = 対照学習後の特徴
        sample['image'] = features
        # sample['adj_s'] = パッチ間のグラフ構造テンソル
        sample['adj_s'] = adj_s     #adj_s.to(torch.double)
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}

        # sample['label'] = ラベル(0,1等)
        # sample['id'] = ptファイルのフォルダ名？(output等)
        # sample['image'] = 特徴データ
        # sample['adj_s'] = グラフ情報のテンソル
        # チェック用
        print("[dataset.py]feature:",features)
        print("[dataset.py]adj_s:",adj_s)

        return sample


    def __len__(self):
        return len(self.ids)