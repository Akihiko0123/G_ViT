import torch
import numpy as np

# simclr.pyより呼び出し
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        # バッチサイズ
        self.batch_size = batch_size
        # 温度
        self.temperature = temperature
        # cudaかcpuかetc.
        self.device = device
        # ソフトマックス
        self.softmax = torch.nn.Softmax(dim=-1)
        # マスク(自己相関マスク)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        # 類似度関数 コサイン類似度
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        # 基準 クロスエントロピー誤差(各サンプルの合計を全体のロスにする[平均(average)もある])
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # コンストラクタから呼び出される
    def _get_similarity_function(self, use_cosine_similarity):
        # コサイン類似度を用いる場合、_cosine_similarityにコサイン類似度(dim=計算する次元数)を入れる
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            # コサイン類似度を返す
            return self._cosine_simililarity
        else:
            # ドット類似度を返す
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # ドット類似度
        # 内積を計算(ノルムで割らない)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        # x = N x C, y = 2N x C
        # "ｘの2次元目の次元追加"(N x 1 x C)と"yの次元追加(1 x 2N x C)"のコサイン類似度
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # コサイン類似度を返す
        return v

    def forward(self, zis, zjs):
        # 予測値zjsとzisを結合
        representations = torch.cat([zjs, zis], dim=0)
        # 類似度行列(自身同士の類似度[コサイン類似度など、指定した類似度]を計算)
        similarity_matrix = self.similarity_function(representations, representations)
        # チェック用
        print(f"similarity_matrix:{similarity_matrix}")

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        print(f"l_pos:{l_pos}")
        print(f"r_pos:{l_pos}")

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        print(f"positives:{positives}")

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        print(f"negatives:{negatives}")

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        print("logits:",logits)

        # 値が0で要素数18(バッチサイズ×2)のtorchをラベルとする
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        print("labels:",labels)
        # logitsと要素18個で値0のtensorでクロスエントロピー誤差を求める
        loss = self.criterion(logits, labels)

        # ロスはバッチサイズ x 2で割った値
        return loss / (2 * self.batch_size)
