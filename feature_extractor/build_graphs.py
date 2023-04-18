import cl as cl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
# OS確認用
import platform

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

# 下記bag_dataset関数から呼び出される
class BagDataset():
    def __init__(self, csv_file, transform=None):
        # パッチのファイルパスリスト
        self.files_list = csv_file
        # 指定の拡張
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        # 1パッチファイルのパス
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        # パッチファイルを開く
        img = Image.open(img)
        # 224 x 224にリサイズ
        img = img.resize((224, 224))
        # sample = 画像の辞書
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        # sample[input]としたらPIL画像が呼ばれる辞書を返す
        return sample 

# 下記bag_dataset関数からBagDataset()クラスで読み込んだ画像(callメソッドで)を元に、
# 画像の拡張処理として呼び出される
class ToTensor(object):
    def __call__(self, sample):
        # 画像取り出し
        img = sample['input']
        # テンソル変換
        img = VF.to_tensor(img)
        return {'input': img} 

# 下記でbag_dataset関数からBag_Datasetクラスへの引数でcallが呼ばれる(拡張が実施される)   
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

# 下記compute_featsから呼び出し
# パッチの座標情報をパッチファイル名称から保存か
def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
#        x, y = path.split('/')[-1].split('.')[0].split('_')
        # windowsでも対応できるように変更
        if platform.system() == "Linux":
            # パッチファイル名を'_'で区切った前半=x,後半=y
            x, y = path.split('/')[-1].split('.')[0].split('_')
        elif platform.system() == "Windows":
            x, y = path.split('\\')[-1].split('.')[0].split('_')        
    txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

# 下記のcompute_featsから呼び出される
# (csv_file_path=パッチファイルパスのリスト、output=積み重ねたバッチごとの特徴データ)
def adj_matrix(csv_file_path, output):
    # パッチの数
    total = len(csv_file_path)
    # パッチ数×パッチ数のnp配列
    adj_s = np.zeros((total, total))

    # パッチの数-1回繰り返す
    for i in range(total-1):
        # path_i = 0個目,1個目,...total-1個目のパッチファイルパス
        path_i = csv_file_path[i]
#        x_i, y_i = path_i.split('/')[-1].split('.')[0].split('_')
        # windowsでも対応できるように変更
        if platform.system() == "Linux":
            x_i, y_i = path_i.split('/')[-1].split('.')[0].split('_')
        elif platform.system() == "Windows":
            # x_i = パッチファイルの"_"を挟んだ前半
            # y_i = パッチファイルの"_"を挟んだ後半
            x_i, y_i = path_i.split('\\')[-1].split('.')[0].split('_')
        ## 0個目と1個目,2個目,...,total個目のパッチ番号(ファイル名)を比較し隣同士かどうかチェック
        ## 1個目と2個目,3個目,...,total個目,...,total-1個目とtotal個目のパッチ番号(ファイル名)を比較し隣同士かどうかチェック
        ## つまり、自分以外の互いのファイルを1回ずつ比較するためにこの二重for文構造になっている。
        # i=0 => 1~total, i=1 => 2~total,...i=total-1 => total-1~total
        for j in range(i+1, total):
            # sptial
            # path_j = 1個目,2個目,...total個目のパッチファイルパス
            path_j = csv_file_path[j]
#            x_j, y_j = path_j.split('/')[-1].split('.')[0].split('_')
            # windowsでも対応できるように変更
            if platform.system() == "Linux":
                x_j, y_j = path_j.split('/')[-1].split('.')[0].split('_')
            elif platform.system() == "Windows":
                # x_j = パッチファイルの"_"を挟んだ前半
                # y_j = パッチファイルの"_"を挟んだ後半
                x_j, y_j = path_j.split('\\')[-1].split('.')[0].split('_')
            # パッチファイル名が順番になっている前提
            # x_iとx_jの差が1以下でy_iとy_jの差が1以下の場合は、adj_s[i][j]とadj_s[j][i]に1を入れる
            # (互いの関係が隣接であることを1で示し、それ以外は0とするnumpy配列)
            # 位置は、同じ列の隣同士か、同じ行の隣同士か、斜め隣りの場合は1になる。
            if abs(int(x_i)-int(x_j)) <=1 and abs(int(y_i)-int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1
    # numpy=>tensor変換
    adj_s = torch.from_numpy(adj_s)
    # cpuにも対応
    adj_s = adj_s.cuda() if torch.cuda.is_available() else adj_s
    # torchの隣接行列adj_sを返す
    # チェック用
    print(f"build_graphs.py_adj_s.shape:{adj_s.shape}")

    return adj_s

    
# 下記compute_feats関数から呼び出される
def bag_dataset(args, csv_file_path):
    # transformed_dataset = 上記BagDatasetクラスのcallメソッドを呼び出した拡張済み画像
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        # 上記ToTensor()クラスのcallメソッドを呼び出すのが拡張処理
                                        ToTensor()
                                    ]))
    # 拡張済み画像のデータローダー
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    # len(transformed_dataset)＝1フォルダ中のパッチの数(?)
    return dataloader, len(transformed_dataset)

# 下記から呼び出し
def compute_feats(args, bags_list, i_classifier, save_path=None, whole_slide_path=None):
    # num_bags = dataset引数のディレクトリの中にある、パッチファイルが入ったフォルダの数
    #            基本1つ？
    num_bags = len(bags_list)
    # チェック用
    print(f"num_bags:{num_bags}")
    Tensor = torch.FloatTensor
    # 0からパッチファイル数まで繰り返し(123個ならi=0~122)
    for i in range(0, num_bags):
        feats_list = []
        # 20倍の場合は.jpegファイルであるという前提で読み込み
        if  args.magnification == '20x':
            # csv_file_path = １フォルダ内のパッチファイルのリスト
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
#            file_name = bags_list[i].split('/')[-3].split('_')[0]
            # Linuxの場合とWindowsの場合で処理を分けるようにアレンジ
            if platform.system() == "Linux":
                file_name = bags_list[i].split('/')[-3].split('_')[0]
            elif platform.system() == "Windows":
                # "output"の部分を取り出す
                file_name = bags_list[i].split('\\')[-3].split('_')[0]
            # Windows用に変更
        # 10倍や5倍の場合は.jpgファイルであるという前提で読み込み
        if args.magnification == '5x' or args.magnification == '10x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
#            file_name = bags_list[i].split('/')[-3].split('_')[0]
            # Linuxの場合とWindowsの場合で処理を分けるようにアレンジ
            if platform.system() == "Linux":
                file_name = bags_list[i].split('/')[-3].split('_')[0]
            elif platform.system() == "Windows":
                # "output"の部分を取り出す
                file_name = bags_list[i].split('\\')[-3].split('_')[0]
        # 上記のbag_dataset関数を呼び出す
        # csv_file_path = １フォルダ内のパッチファイルパスのリスト
        # dataloader=拡張処理付(テンソル変換)データローダー、bag_size=パッチの数
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        print('{} files to be processed: {}'.format(len(csv_file_path), file_name))
        
        if os.path.isdir(os.path.join(save_path, 'simclr_files', file_name)) or len(csv_file_path) < 1:
            print('already exists')
            continue
        with torch.no_grad():
            # データローダーからバッチごとに繰り返し
            for iteration, batch in enumerate(dataloader):
                # cpuにも対応
                # batch["input"]でパッチ単位で画像を取り出す
                patches = batch['input'].float().cuda() if torch.cuda.is_available() else batch['input'].float()                
                # i_classifierインスタンス
                # feats = feats.view(feats.shape[0],-1) = 特徴層からの出力を0次元目を保持して二次元に変換したテンソル
                # classes = resnet出力からFCLでクラス数を変更した出力結果(テンソル形式)
                # i_classifierクラスのcall()を呼びバッチ単位の推論を実行
                feats, classes = i_classifier(patches)
                #feats = feats.cpu().numpy()
                # 特徴をfeats_listに加える
                feats_list.extend(feats)
        
        '''
        上記までは、対照学習で学習したモデルを使った推論処理
        '''
        os.makedirs(os.path.join(save_path, 'simclr_files', file_name), exist_ok=True)

        txt_file = open(os.path.join(save_path, 'simclr_files', file_name, 'c_idx.txt'), "w+")
        # 上記save_coords関数を呼びだし
        # パッチの座標情報をtxtファイル内に記録(上書きなので恐らく最後に処理したパッチのファイル名("_"で区切った前後半)だけが記載される)
        save_coords(txt_file, csv_file_path)
        # save node features
        # 複数あるバッチごとの特徴を0次元目を軸に積み重ねる形にする(0次元目がバッチ数になる？)
        # 参考：https://panda-clip.com/torch-stack/
        output = torch.stack(feats_list, dim=0).cuda() if torch.cuda.is_available() else torch.stack(feats_list, dim=0)
        torch.save(output, os.path.join(save_path, 'simclr_files', file_name, 'features.pt'))

        
        # save adjacent matrix
        # バッチごとに特徴を0次元目で積み重ねたデータとパッチファイルパスのリスト、を基に
        # 上記adj_matrix関数を呼び出す
        # adj_s＝隣同士の情報が入ったtensor
        adj_s = adj_matrix(csv_file_path, output)
        # ptファイルとして保存する
        torch.save(adj_s, os.path.join(save_path, 'simclr_files', file_name, 'adj_s.pt'))

        print('\r Computed: {}/{}'.format(i+1, num_bags))
        
# コマンド実行直後に呼ばれる
def main():
    ## 引数
    # WSIの特徴をSimCLR embedder(特徴抽出器の学習で得たモデル？)で計算する
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    # 出力クラス数
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    # 特徴数
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    # バッチ数(データローダーで出力される)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    # 作業プロセスの数(PCの場合は1つで良いはず)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    # パッチディレクトリが入ったディレクトリへのパス
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    # バックボーン(層の前半～後半の構造)の指定(デフォルト=resnet18)
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    # 拡大レベル(デフォルト=20倍)
    parser.add_argument('--magnification', default='20x', type=str, help='Magnification to compute features')
    # 事前学習の重みデータ(モデル)へのパス(デフォルト=None)
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    # グラフフォルダの出力先ディレクトリパス(デフォルト=None)
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()
    
    # バックボーンがresnet18の場合は、torchvision.modelsから
    # モデルを読み込む。事前学習はなし、正規化手法は2dインスタンス正規化を指定
    # (バッチ数が少ない場合や、NLPのように学習データよりもテストデータが長い可能性がある場合に学習が安定しないため、
    # NLPではLayer正規化が用いられる。しかし、Layer正規化はチャンネル方向にもまとめて正規化する。
    # 画像はRGBの3CHあるので全てをまとめて正規化するよりも各々分けて正規化する
    # instance正規化の方が適切という判断がされたと思われる。
    # インスタンス正規化ではバッチ単位でもなく、１枚のパッチ毎に平均・分散を計算する)
    if args.backbone == 'resnet18':
#        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        # torch_visionのバージョン0.15からはpretrainedオプション(pretrained=True)が
        # weightsオプションでweights = 'ResNet50_Weights.DEFAULT'などで直接指定する形に
        # 正式に変更となりpretrainedオプションはエラーになる。
        # 一応対応としてコードを書き替える
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    # 全パラメータを更新しないように設定
    for param in resnet.parameters():
        param.requires_grad = False
    # resnet.fc = Skip connection用に指定した時点の形状を保持するためのインスタンス？
    #             (後でreshapeなどで形状が変更されていくはずだが、変更前の形状を保持)
    resnet.fc = nn.Identity()
    # i_classifier = cl.pyのIClassifierクラスのインスタンス
    # cpuで動くように一部アレンジ
    i_classifier = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()\
    if torch.cuda.is_available() else cl.IClassifier(resnet, num_feats, output_class=args.num_classes)
    
    # load feature extractor
    if args.weights is None:
        print('No feature extractor')
        return
    # 対照学習の重み辞書(各層の重み情報が入った辞書)を読み込み(出力次元=512)
    state_dict_weights = torch.load(args.weights)
    # state_dict_init = IClassifierクラスのインスタンスのstate_dict()メソッド
    state_dict_init = i_classifier.state_dict()
    # チェック用
#    print(state_dict_init["fc.weight"].shape)
    # 順序付辞書を用意(python3.7以降は通常の辞書でも順序を記憶するが、そちらでも良い?)
    # ➡対照学習時のモデルの形式がordered_dictなので、通常の辞書だとエラーになると思われる
    new_state_dict = OrderedDict()
    # 読み込んだresnetの事前学習なしの重みを、対照学習で学んだ重みに置き換える作業
    # つまり、出力形状も一致しないとエラーとなる。
    # k=対照学習した重みのキー、v=対照学習した重みの値
    # k_0=学習していない重みのキー、v_0=学習していない重みの値
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        # 名前は学習していない器となるモデルをベースにする
        name = k_0
        # 対照学習済みの重みを学習していないモデルの重みに代入していく
        new_state_dict[name] = v
    # 本モデルに合わせて置き換えた重みを分類器の重みとして読み込み
    # strict=False => マッチしたキーだけを読み込み
    i_classifier.load_state_dict(new_state_dict, strict=False)
 
    os.makedirs(args.output, exist_ok=True)
    # パッチファイルが置かれたディレクトリ内のファイル名を取り出し
    bags_list = glob.glob(args.dataset)
    
    # 上記のcompute_feats関数を呼び出し
    compute_feats(args, bags_list, i_classifier, args.output)
    
# コマンド実行時にスタート
if __name__ == '__main__':
    # main関数を実行
    main()
