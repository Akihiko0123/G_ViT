'''
    File name: tile_WSI.py
    Date created: March/2021
	Source:
	Tiling code inspired from
	https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py

	The code has been extensively modified 
	Objective:
	Tile svs, jpg or dcm images with the possibility of rejecting some tiles based based on xml or jpg masks
	Be careful:
	Overload of the node - may have memory issue if node is shared with other jobs.
'''

from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
try:
   import pydicom as dicom
except ImportError:
   import dicom
# from scipy.misc import imsave
from imageio import imwrite as imsave
# from scipy.misc import imread
from imageio import imread
# from scipy.misc import imresize

from xml.dom import minidom
from PIL import Image, ImageDraw, ImageCms
from skimage import color, io
# PILで開く画像のピクセル数制限を解除
Image.MAX_IMAGE_PIXELS = None

# スライド名
VIEWER_SLIDE_NAME = 'slide'

# DeepZoomStaticTilerのコンストラクタからrun()メソッドが呼び出される。workerの数だけ繰り返し実行する
# (self._queue, slidepath, tile_size, overlap,limit_bounds, quality, self._Bkg, self._ROIpc)
class TileWorker(Process):
    # タイルを作成する子プロセス
    """A child process that generates and writes tiles."""
    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,quality, _Bkg, _ROIpc):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._Bkg = _Bkg
        self._ROIpc = _ROIpc
    # 色空間をRGBからlab（人間の視覚に近いLとaとbから成る色空間）に変換するメソッド()
    def RGB_to_lab(self, tile):
        # srgb_p = ImageCms.createProfile("sRGB")
        # lab_p  = ImageCms.createProfile("LAB")
        # rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        # Lab = ImageCms.applyTransform(tile, rgb2lab)
        # Lab = np.array(Lab)
        # Lab = Lab.astype('float')
        # Lab[:,:,0] = Lab[:,:,0] / 2.55
        # Lab[:,:,1] = Lab[:,:,1] - 128
        # Lab[:,:,2] = Lab[:,:,2] - 128
        print("RGB to Lab")
        Lab = color.rgb2lab(tile)
        return Lab
    # LabからRGBに変換するメソッド
    def Lab_to_RGB(self,Lab):
        # srgb_p = ImageCms.createProfile("sRGB")
        # lab_p  = ImageCms.createProfile("LAB")
        # lab2rgb = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "LAB", "RGB")
        # Lab[:,:,0] = Lab[:,:,0] * 2.55
        # Lab[:,:,1] = Lab[:,:,1] + 128
        # Lab[:,:,2] = Lab[:,:,2] + 128
        # newtile = ImageCms.applyTransform(Lab, lab2rgb)
        print("Lab to RGB")
        newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
        return newtile

    # タイルの標準化
    def normalize_tile(self, tile, NormVec):
        Lab = self.RGB_to_lab(tile)
        TileMean = [0,0,0]
        TileStd = [1,1,1]
        newMean = NormVec[0:3] 
        newStd = NormVec[3:6]
        for i in range(3):
            TileMean[i] = np.mean(Lab[:,:,i])
            TileStd[i] = np.std(Lab[:,:,i])
            # print("mean/std chanel " + str(i) + ": " + str(TileMean[i]) + " / " + str(TileStd[i]))
            tmp = ((Lab[:,:,i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
            if i == 0:
                tmp[tmp<0] = 0 
                tmp[tmp>100] = 100 
                Lab[:,:,i] = tmp
            else:
                tmp[tmp<-128] = 128 
                tmp[tmp>127] = 127 
                Lab[:,:,i] = tmp
        tile = self.Lab_to_RGB(Lab)
        return tile
    # タイル毎に背景の判定やマスクの判定・保存などを実施
    def run(self):
        # slideパスからスライドを読み込みself._slideに入れる
        self._slide = open_slide(self._slidepath)
        # last_associated 変数を Noneにする
        last_associated = None
        # dz = _get_dz()メソッドで処理されDeepZoomGeneratorメソッドで返される
        #      タイルオブジェクト＝タイルのサイズとオーバーラップは指定された状態のオブジェクト
        #                         (スライド(wsi)データから指定解像度レベル(1~16)で指定場所(横x番目,縦y番目)のタイルを生成できる)
        dz = self._get_dz()
        while True:
            # _queue.get()
            # キューからデータを持ってくる
            # （(self._associated, level, (col, row),tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask, self._normalize)）等？
            data = self._queue.get()
            if data is None:
                # dataが空なら終了してループから抜ける
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            # dataから各種引数や変数を取りだす
            associated, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask, Normalize = data
            # last_associated != associatedなら、引数のassociatedに合わせてスライドオブジェクトを取得しdzとする
            # その後、last_associatedにassociatedを代入する。
            # つまり、スライドが切り替わるタイミングでassociatedが変わるということか？
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            if True:
                try:
                    # tile = サイズやオーバーラップを指定されたスライド内タイルオブジェクトの指定解像度、場所(x,y)のパッチ
                    tile = dz.get_tile(level, address)
                    # A single tile is being read
                    #check the percentage of the image with "information". Should be above 50%
                    # [L = R * 299/1000 + G * 587/1000 + B * 114/1000]の輝度でグレースケールに変換
                    gray = tile.convert('L')
                    # 輝度=xの大きさで二値化している(32ビットfloatピクセルで表現)
                    bw = gray.point(lambda x: 0 if x<220 else 1, 'F')
                    # numpy配列
                    arr = np.array(np.asarray(bw))
                    # bwの平均値
                    avgBkg = np.average(bw)
                    # 1ビット("白"or"黒")を作成
                    bw = gray.point(lambda x: 0 if x<220 else 1, '1')
                    # check if the image is mostly background
                    # 2値化した結果背景が多いかどうかを確認(1[白] or 0[黒]の平均値が一定_Bkgより大きければ白が多く背景でないと判断)
                    # _Bkg= プログラム実行引数、デフォルトは50。つまり, タイル全体の1[白] or 0[黒]の平均値が0.5以下の場合は背景タイルだと
                    # 　　　　みなされる。
                    print("res: " + outfile + " is " + str(avgBkg))
                    # もしも平均ピクセルがself._Bkg(デフォルト50/100=0.5)以下の場合
                    if avgBkg <= (self._Bkg / 100.0):
                        # print("PercentMasked: %.6f, %.6f" % (PercentMasked, self._ROIpc / 100.0) )
                        # if an Aperio selection was made, check if is within the selected region
                        # もしもPercentMasked(マスク割合(getでデータから取得？_write_tilesメソッドで実行))
                        # がROI(対象領域)値[デフォルト50/100 = 0.5]以上の場合、
                        if PercentMasked >= (self._ROIpc / 100.0):

                            if Normalize != '':
                                print("normalize " + str(outfile))
                                # arrtile = np.array(tile)
                                # タイルをnormalize_tileメソッドで標準化して配列から画像データへ変換
                                tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype('uint8'),'RGB')

                            tile.save(outfile, quality=self._quality)
                            if bool(SaveMasks)==True:
                                height = TileMask.shape[0]
                                width = TileMask.shape[1]
                                TileMaskO = np.zeros((height,width,3), 'uint8')
                                maxVal = float(TileMask.max())
                                TileMaskO[...,0] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,1] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,2] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO = numpy.array(Image.fromarray(TileMaskO).resize(arr.shape[0], arr.shape[1],3))
                                # TileMaskO = imresize(TileMaskO, (arr.shape[0], arr.shape[1],3))
                                TileMaskO[TileMaskO<10] = 0
                                TileMaskO[TileMaskO>=10] = 255
                                imsave(outfile_bw,TileMaskO) #(outfile_bw, quality=self._quality)

                        #print("%s good: %f" %(outfile, avgBkg))
                    #elif level>5:
                    #    tile.save(outfile, quality=self._quality)
                            #print("%s empty: %f" %(outfile, avgBkg))
                    self._queue.task_done()
                except Exception as e:
                    # print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    # e = sys.exc_info()[0]
                    print(e)
                    self._queue.task_done()
    # runメソッドから呼び出される
    def _get_dz(self, associated=None):
        if associated is not None:
            # associatedがNoneでない場合は、openslideで指定イメージを読み込む
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            # associatedでなければ、既に読み込んでいるスライドをimageに入れる
            image = self._slide
        # DeepZoomGenerator=スライドをタイルに切り分けたGeneratorオブジェクトを取得するメソッド
        # Generatorオブジェクトもまたopenslideで開くスライド同様に複数の解像度レベルの情報を保有している。
        # レベルにはスライド以上の段階がある。16倍レベル程のレベルを持っている。(オブジェクト.tiles_countで確認可能)
        # なお、スライドオブジェクトではレベル0が一番大きいサイズだったが、
        # タイルオブジェクトでは、レベルの数値が大きくなるほどサイズも大きい。(16がレベル0に相当？)
        # 仮に(32000,38400)から(1000,1200)サイズにダウンサンプリングする場合で、tile_sizeが256の場合、
        # オーバーラップがないならタイルは幅方向に4枚程度(3枚＋少し小さいタイル1枚?)になる。(1000/256=3.90625)
        # 高さ方向は5枚程度(4枚＋少し小さいタイル1枚？)になり、全体のshapeは4x5になる。
        # つまり、全体のタイル数は20枚、各サイズは_tile_size(割り切れない1枚だけ小さいタイル)
        # (オブジェクト.level_tiles[level_num(11?)]で確認可能)
        # 各タイルの大きさは、オブジェクト.get_tile_dimension(level_num(11?), (横タイル番号,縦タイル番号))で確認できる
        # タイル番号は0番目からスタート＝＞4,5の位置を見たい場合は3,4と指定
        # タイルオブジェクトを16レベルで(62番目, 70番目)の位置に来る１枚のタイルの情報を取得するコマンドは
        # single_tile = オブジェクト.get_tile(16, (62, 70))となる。
        # 更に、rgbに変換するには、single_tile_RGB = single_tile.convert('RGB')で可能。(RGBAのAを除去する場合)
        # 表示はsingle_tile_RGB.show()
        # limit_bounds = Trueの場合、空のスライドは処理しないと思われる
        # (参考:https://www.youtube.com/watch?v=QntLBvUZR5c）
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)

# 683行目位のDeepZoomStaticTilerの_run_image()メソッドより呼び出される
class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""
    # 下記からよびだされた際の各変数
    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, xmlfile, mask_type, xmlLabel, ROIpc, ImgExtension, SaveMasks, Mag, normalize, Fieldxml):
        self._dz = dz                  
        self._basename = basename      
        self._basenameJPG = basenameJPG
        self._format = format          
        self._associated = associated  
        self._queue = queue            
        self._processed = 0            
        self._slide = slide
        self._xmlfile = xmlfile         
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml
    # DeepZoomStaticTilerの_run_image()メソッドより呼び出される
    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
            ########################################3
            # nc_added
        #level = self._dz.level_count-1
        Magnification = 20
        tol = 2
        #get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            # print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            print(self._ImgExtension)
            if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension) | ("tif" in self._ImgExtension):
                #Objective = self._ROIpc
                Objective = 1.
                Magnification = Objective
                print("input is jpg - will be tiled as such with %f" % Objective)
            else:
                return
        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
          print(self._basename + " - Objective field empty!")
          return
        '''
        if(min(AbsMismatch) <= tol):
            Level = int(AbsMismatch.index(min(AbsMismatch)))
            Factor = 1
        else: #pick next highest level, downsample
            Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
            Factor = Magnification / Available[Level]
        # end added
        '''
        xml_valid = False
        # a dir was provided for xml files

        '''
        ImgID = os.path.basename(self._basename)
        Nbr_of_masks = 0
        if self._xmlfile != '':
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            print("xml:")
            print(xmldir)
            if os.path.isfile(xmldir):
                xml_labels, xml_valid = self.xml_read_labels(xmldir)
                Nbr_of_masks = len(xml_labels)
            else:
                print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (ImgID, xmldir) )
                return
        else:
            Nbr_of_masks = 1
        '''

        if True:
            #if self._xmlfile != '' && :
            # print(self._xmlfile, self._ImgExtension)
            ImgID = os.path.basename(self._basename)
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            # print("xml:")
            # print(xmldir)
            if (self._xmlfile != '') & (self._ImgExtension != 'jpg') & (self._ImgExtension != 'dcm'):
                # print("read xml file...")
                mask, xml_valid, Img_Fact = self.xml_read(xmldir, self._xmlLabel, self._Fieldxml)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return
            elif (self._xmlfile != '')  & (self._ImgExtension == 'dcm'):
                # print("check mask for dcm")
                mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
                # mask <-- read mask 
                #  Img_Fact <-- 1
                # xml_valid <-- True if mask file exists.
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return

            # print("current directory: %s" % self._basename)

            #return
            #print(self._dz.level_count)

            for level in range(self._dz.level_count-1,-1,-1):
                ThisMag = Available[0]/pow(2,self._dz.level_count-(level+1))
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue
                ########################################
                #tiledir = os.path.join("%s_files" % self._basename, str(level))
                tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                cols, rows = self._dz.level_tiles[level]
                if xml_valid:
                    # print("xml valid")
                    '''# If xml file is used, check for each tile what are their corresponding coordinate in the base image
                    IndX_orig, IndY_orig = self._dz.level_tiles[-1]
                    CurrentLevel_ReductionFactor = (Img_Fact * float(self._dz.level_dimensions[-1][0]) / float(self._dz.level_dimensions[level][0]))
                    startIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
                    print("***********")
                    endIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
                    endIndX_current_level_conv.append(self._dz.level_dimensions[level][0])
                    endIndX_current_level_conv.pop(0)
    
                    startIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
                    #endIndX_current_level_conv = [i * CurrentLevel_ReductionFactor - 1 for i in range(rows)]
                    endIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
                    endIndY_current_level_conv.append(self._dz.level_dimensions[level][1])
                    endIndY_current_level_conv.pop(0)
                    '''
                    #startIndY_current_level_conv = []
                    #endIndY_current_level_conv = []
                    #startIndX_current_level_conv = []
                    #endIndX_current_level_conv = []

                    #for row in range(rows):
                    #    for col in range(cols):
                    #        Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level,(col, row))
                    #        Ddimension = self._dz.get_tile_dimensions(level,(col, row))
                    #        startIndY_current_level_conv.append(int((Dlocation[1]) / Img_Fact))
                    #        endIndY_current_level_conv.append(int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                    #        startIndX_current_level_conv.append(int((Dlocation[0]) / Img_Fact))
                    #        endIndX_current_level_conv.append(int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            # print(Dlocation, Ddimension, int((Dlocation[1]) / Img_Fact), int((Dlocation[1] + Ddimension[1]) / Img_Fact), int((Dlocation[0]) / Img_Fact), int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                          tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                          self._basenameJPG, col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                          self._basenameJPG, col, row, self._format))
                        else:
                          tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                          col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                          col, row, self._format))
                        if xml_valid:
                            # compute percentage of tile in mask
                            # print(row, col)
                            # print(startIndX_current_level_conv[col])
                            # print(endIndX_current_level_conv[col])
                            # print(startIndY_current_level_conv[row])
                            # print(endIndY_current_level_conv[row])
                            # print(mask.shape)
                            # print(mask[startIndX_current_level_conv[col]:endIndX_current_level_conv[col], startIndY_current_level_conv[row]:endIndY_current_level_conv[row]])
                            # TileMask = mask[startIndY_current_level_conv[row]:endIndY_current_level_conv[row], startIndX_current_level_conv[col]:endIndX_current_level_conv[col]]
                            # PercentMasked = mask[startIndY_current_level_conv[row]:endIndY_current_level_conv[row], startIndX_current_level_conv[col]:endIndX_current_level_conv[col]].mean() 
                            # print(startIndY_current_level_conv[row], endIndY_current_level_conv[row], startIndX_current_level_conv[col], endIndX_current_level_conv[col])

                            Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level,(col, row))
                            Ddimension = tuple([pow(2,(self._dz.level_count - 1 - level)) * x for x in self._dz.get_tile_dimensions(level,(col, row))])
                            startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            # print(Ddimension, Dlocation, Dlevel, Dsize, self._dz.level_count , level, col, row)

                            #startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            #endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            #startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            #endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            TileMask = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv]
                            PercentMasked = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv].mean() 

                            # print(Ddimension, startIndY_current_level_conv, endIndY_current_level_conv, startIndX_current_level_conv, endIndX_current_level_conv)


                            if self._mask_type == 0:
                                # keep ROI outside of the mask
                                PercentMasked = 1.0 - PercentMasked
                                # print("Invert Mask percentage")

                            # if PercentMasked > 0:
                            #     print("PercentMasked_p %.3f" % (PercentMasked))
                            # else:
                            #     print("PercentMasked_0 %.3f" % (PercentMasked))

 
                        else:
                            PercentMasked = 1.0
                            TileMask = []

                        if not os.path.exists(tilename):
                            # キューに指定のデータを入れる
                            self._queue.put((self._associated, level, (col, row),
                                        tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask, self._normalize))
                        self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


    def jpg_mask_read(self, xmldir):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]
        # Img_Fact = int(ImgMaxSizeX_orig / 1.0 / cols)
        Img_Fact = 1
        try:
            # xmldir: change extension from xml to *jpg   
            xmldir = xmldir[:-4] + "mask.jpg"
            # xmlcontent = read xmldir image
            xmlcontent = imread(xmldir)
            xmlcontent = xmlcontent - np.min(xmlcontent)
            mask = xmlcontent / np.max(xmlcontent)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        return mask, xml_valid, Img_Fact


    def xml_read(self, xmldir, Attribute_Name, Fieldxml):

        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]

        NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig),15000.0)
        # Img_Fact = 
        # read_region(location, level, size)
        # dz.get_tile_coordinates(14,(0,2))
        # ((0, 1792), 1, (320, 384))

        Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)
       
        # print("image info:")
        # print(ImgMaxSizeX_orig, ImgMaxSizeY_orig, cols, rows) 
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        xy = {}
        xy_neg = {}
        NbRg = 0
        labelIDs = xmlcontent.getElementsByTagName('Annotation')
        # print("%d labels" % len(labelIDs) )
        for labelID in labelIDs:
            if (Attribute_Name==[]) | (Attribute_Name==''):
                    isLabelOK = True
            else:
                try:
                    labeltag = labelID.getElementsByTagName('Attribute')[0]
                    if (Attribute_Name==labeltag.attributes[Fieldxml].value):
                    # if (Attribute_Name==labeltag.attributes['Value'].value):
                    # if (Attribute_Name==labeltag.attributes['Name'].value):
                        isLabelOK = True
                    else:
                        isLabelOK = False
                except:
                	isLabelOK = False
            if Attribute_Name == "non_selected_regions":
                isLabelOK = True

            #print("label ID, tag:")
            #print(labelID, Attribute_Name, labeltag.attributes['Name'].value)
            #if Attribute_Name==labeltag.attributes['Name'].value:
            if isLabelOK:
                regionlist = labelID.getElementsByTagName('Region')
                for region in regionlist:
                    vertices = region.getElementsByTagName('Vertex')
                    NbRg += 1
                    regionID = region.attributes['Id'].value + str(NbRg)
                    NegativeROA = region.attributes['NegativeROA'].value
                    # print("%d vertices" % len(vertices))
                    if len(vertices) > 0:
                        #print( len(vertices) )
                        if NegativeROA=="0":
                            xy[regionID] = []
                            for vertex in vertices:
                                # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                # x = int(round(float(vertex.attributes['X'].value) / ImgMaxSizeX_orig * (cols*Img_Fact)))
                                # y = int(round(float(vertex.attributes['Y'].value) / ImgMaxSizeY_orig * (rows*Img_Fact)))
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy[regionID].append((x,y))
                                #print(vertex.attributes['X'].value, vertex.attributes['Y'].value, x, y )
    
                        elif NegativeROA=="1":
                            xy_neg[regionID] = []
                            for vertex in vertices:
                                # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                # x = int(round(float(vertex.attributes['X'].value) / ImgMaxSizeX_orig * (cols*Img_Fact)))
                                # y = int(round(float(vertex.attributes['Y'].value) / ImgMaxSizeY_orig * (rows*Img_Fact)))
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy_neg[regionID].append((x,y))
    

                        #xy_a = np.array(xy[regionID])

        # print("%d xy" % len(xy))
        #print(xy)
        # print("%d xy_neg"  % len(xy_neg))
        #print(xy_neg)
        # print("Img_Fact:")
        # print(NewFact)
        # img = Image.new('L', (int(cols*Img_Fact), int(rows*Img_Fact)), 0)
        img = Image.new('L', (int(ImgMaxSizeX_orig/NewFact), int(ImgMaxSizeY_orig/NewFact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=0)
        #img = img.resize((cols,rows), Image.ANTIALIAS)
        mask = np.array(img)
        #print(mask.shape)
        if Attribute_Name == "non_selected_regions":
        	# scipy.misc.toimage(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
                Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
        else:
           if self._mask_type==0:
               # scipy.misc.toimage(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
               Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
           else:
               # scipy.misc.toimage(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
               Image.fromarray(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))  
        #print(mask)
        return mask / 255.0, xml_valid, NewFact
        # Img_Fact

# 本プログラム実行後に、下記960行目辺りで本クラスのrunメソッドが呼び出される。
# スライド内の全画像のタイルとメタデータの生成を処理
class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""
# (filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, 
#  opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml)
    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG, xmlfile, mask_type, ROIpc, oLabel, ImgExtension, SaveMasks, Mag, normalize, Fieldxml):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        #print("line226 - %s " % (slidepath) )
        self._slide = open_slide(slidepath) # open_slide(filename)
        self._basename = basename           # output
        self._basenameJPG = basenameJPG     # opts.basenameJPG
        self._xmlfile = xmlfile             # opts.xmlfile
        self._mask_type = mask_type         # opts.mask_type
        self._format = format               # opts.format
        self._tile_size = tile_size         # opts.tile_size
        self._overlap = overlap             # opts.overlap
        self._limit_bounds = limit_bounds   # opts.limit_bounds
        self._queue = JoinableQueue(2 * workers)   # JoinableQueue(2 * opts.workers) 処理数？
        self._workers = workers             # opts.workers
        self._with_viewer = with_viewer     # opts.with_viewer
        self._Bkg = Bkg                     # opts.Bkg
        self._ROIpc = ROIpc                 # opts.ROIpc
        self._dzi_data = {}                 
        self._xmlLabel = oLabel             # ''
        self._ImgExtension = ImgExtension   # ImgExtension
        self._SaveMasks = SaveMasks         # opts.SaveMasks
        self._Mag = Mag                     # opts.Mag
        self._normalize = normalize         # opts.normalize
        self._Fieldxml = Fieldxml           # opts.Fieldxml
        # workersの数だけTileWorkerのstartメソッドを実行
        # startメソッドはthredingの手法で、マルチスレッドの処理に用いられる。
        # 実際はstart()メソッドを用意するのではなく、各workerでrun()メソッドが実行される事になる。
        for _i in range(workers):
            # 標準化した上で、各タイルやタイルマスクを作成・出力フォルダに保存する
            TileWorker(self._queue, slidepath, tile_size, overlap,
                limit_bounds, quality, self._Bkg, self._ROIpc).start()

    def run(self):
        # _run_imageメソッドを実行
        self._run_image()
        # _with_viewer変数が設定されている場合、
        if self._with_viewer:
            # _slide.associated_imagesの枚数だけ繰り返し
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()
    # _slide変数から1枚の画像を処理
    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        # associated がNoneの場合、Imageは_slideで_with_viewerがTrueなら
        if associated is None:
            image = self._slide
            if self._with_viewer:
                 basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        # print("enter DeepZoomGenerator")
        # dz = 指定したサイズ、オーバーラップのタイルオブジェクト
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,limit_bounds=self._limit_bounds)
        # print("enter DeepZoomImageTiler")
        # DeepZoomImageTilerのオブジェクトを作成
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,self._queue, self._slide, self._basenameJPG, self._xmlfile, self._mask_type, self._xmlLabel, self._ROIpc, self._ImgExtension, self._SaveMasks, self._Mag, self._normalize, self._Fieldxml)
        # DeepZoomImageTilerクラスのオブジェクトtilerについて、run()メソッドを実行
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()



    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__),autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        # Embed the dzi metadata in the HTML to work around Chrome's
        # refusal to allow XmlHttpRequest from file:///, even when
        # the originating page is also a file:///
        data = template.render(slide_url=self._url_for(None),slide_mpp=mpp,associated=associated_urls, properties=self._slide.properties, dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()



def ImgWorker(queue):
	# print("ImgWorker started")
	while True:
		cmd = queue.get()			
		if cmd is None:
			queue.task_done()
			break
		# print("Execute: %s" % (cmd))
		subprocess.Popen(cmd, shell=True).wait()
		queue.task_done()

def xml_read_labels(xmldir, Fieldxml):
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid
        labeltag = xmlcontent.getElementsByTagName('Attribute')
        xml_labels = []
        for xmllabel in labeltag:
            xml_labels.append(xmllabel.attributes[Fieldxml].value)
            #xml_labels.append(xmllabel.attributes['Name'].value)
            # xml_labels.append(xmllabel.attributes['Value'].value)
        if xml_labels==[]:
            xml_labels = ['']
        # print(xml_labels)
        return xml_labels, xml_valid 

# ここから実行
if __name__ == '__main__':
    # OptionParser: argparseの古いバージョンであり、廃止予定
	parser = OptionParser(usage='Usage: %prog [options] <slide>')
    # -L：境界無視、limit_boundsという名前で呼ばれる, 全体スキャン領域を表示する
	parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
		default=True, action='store_false',
		help='display entire scan area')
    # -e：オーバーラップ、隣接タイルとの重なり
    #     (タイルの各内側のエッジに追加する余分なピクセル数として定義されている。
    #      つまり、最終的にタイルサイズはs + 2.eとなる。
    #      そのため、512pxのタイルを50%オーバーラップしたければs(サイズ引数)を256にして、eを128に設定する必要がある)
    #      この場合256 + 2 * 128 = 512が全体サイズとなり、全体で50%オーバーラップしたことになる。
	parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
		type='int', default=1,
		help='overlap of adjacent tiles [1]')
    # -f：タイルに分ける画像のフォーマット
	parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
		default='jpeg',
		help='image format for tiles [jpeg]')
    # -j：実行するワーカープロセスの数
	parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
		type='int', default=4,
		help='number of worker processes to start [4]')
    # -o：出力ファイル名
	parser.add_option('-o', '--output', metavar='NAME', dest='basename',
		help='base name of output file')
    # -Q：JPEGの圧縮品質
	parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
		type='int', default=90,
		help='JPEG compression quality [90]')
    # -r：ディレクトリツリーとHTMLビュアーを生成
	parser.add_option('-r', '--viewer', dest='with_viewer',
		action='store_true',
		help='generate directory tree with HTML viewer')
    # -s：タイルサイズ(-eとの組み合わせで変わる可能性あり)
	parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
		type='int', default=254,
		help='tile size [254]')
    # -B：残す背景のしきい値(%)[デフォルト50%]　二値化で黒色になるピクセルが50%を超えているパッチは背景として除外？
	parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',
		type='float', default=50,
		help='Max background threshold [50]; percentager of background allowed')
    # -x：必要な場合に指定するxmlファイル
	parser.add_option('-x', '--xmlfile', metavar='NAME', dest='xmlfile',
		help='xml file if needed')
    # -F：どのxmlファイルのフィールドでラベルが保存されたのか
	parser.add_option('-F', '--Fieldxml', metavar='{Name|Value}', dest='Fieldxml',
		default='Value',
		help='which field of the xml file is the label saved')
    # -m：xmlファイルが使用される場合、タイルをROI(関心領域)に入れるか(1)、ROI外にするか(0)
	parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type',
		type='int', default=1,
		help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
    # -R：xmlと一緒に用いられる。ROI(white)にカバーされるタイルの最小パーセンテージ
	parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc',
		type='float', default=50,
		help='To be used with xml file - minimum percentage of tile covered by ROI (white)')
    # -l：xmlと一緒に用いられる。oLabelに含まれる文字を含むラベルのみタイルする。
	parser.add_option('-l', '--oLabelref', metavar='NAME', dest='oLabelref',
		help='To be used with xml file - Only tile for label which contains the characters in oLabel')
    # -S：yesにセットすることで、全てのタイルの全てのマスクを保存する(maskという接頭辞で同じディレクトリに保存される)
	parser.add_option('-S', '--SaveMasks', metavar='NAME', dest='SaveMasks',
		default=False,
		help='set to yes if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')
    # -t：出力フォルダのベースネーム(拡張子抜きのフォルダ名、中間dcm画像を保存するためのもの)
	parser.add_option('-t', '--tmp_dcm', metavar='NAME', dest='tmp_dcm',
		help='base name of output folder to save intermediate dcm images converted to jpg (we assume the patient ID is the folder name in which the dcm images are originally saved)')
    # -M：タイリングを行う倍率 (-1倍)
	parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
		type='float', default=-1,
		help='Magnification at which tiling should be done (-1 of all)')
    # -N：正規化が必要な場合、Nは各チャンネルの平均と標準偏差のリストを表す。
    #     例えば、\'57,22,-8,20,10,5\'最初の3つの数字はターゲットの平均値、後の3つの数字はターゲットの標準偏差になる
    #     例えば、RGBの3チャンネルなら6つの数字が指定されていればOK
	parser.add_option('-N', '--normalize', metavar='NAME', dest='normalize',
		help='if normalization is needed, N list the mean and std for each channel. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds')




	(opts, args) = parser.parse_args()


	try:
        # スライドのパスは1つ目の引数
		slidepath = args[0]
	except IndexError:
        # ない場合はスライド引数がありませんと表示する
		parser.error('Missing slide argument')
    # basename引数(dest=basename)がない場合、つまり出力フォルダを表す-o引数が指定されていない場合は
    # slidepathのベースネーム(パス部分を除いたファイル名部分)を使用する
	if opts.basename is None:
		opts.basename = os.path.splitext(os.path.basename(slidepath))[0]
    # xmlfileが指定されていない場合
	if opts.xmlfile is None:
		opts.xmlfile = ''

	try:
        # normalize(正規化)が指定されている場合
		if opts.normalize is not None:
            # normalize＝normalize引数を","で分けて各々小数変換し、リスト化したもの
			opts.normalize = [float(x) for x in opts.normalize.split(',')]
            # もしも長さが6個でない場合''として、正規化が適用されなかったエラーを出力
			if len(opts.normalize) != 6:
				opts.normalize = ''
				parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right length - 6 values expected")
		else:
			opts.normalize  = ''
    # normalizeが指定されていなければ正規化が適用されなかったエラーを出力
	except:
		opts.normalize = ''
		parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right format")
        #if ss != '':
        #    if os.path.isdir(opts.xmlfile):
            

	# Initialization
	# imgExample = "/ifs/home/coudrn01/NN/Lung/RawImages/*/*svs"
	# tile_size = 512
	# max_number_processes = 10
	# NbrCPU = 4

	# get  images from the data/ file.
    # files=slidepath内のWSIファイル名のリスト
	files = glob(slidepath)
	#ImgExtension = os.path.splitext(slidepath)[1]
    # "パス\\*.png"などでファイルを取り出す前提
    # パスを分割して後ろを取り出すので、拡張子(.png等)を取り出すことになる。
	ImgExtension = slidepath.split('*')[-1]
	#files
	#len(files)
	# print(args)
	# print(args[0])
	# print(slidepath)
	# print(files)
	# print("***********************")

	'''
	dz_queue = JoinableQueue()
	procs = []
	print("Nb of processes:")
	print(opts.max_number_processes)
	for i in range(opts.max_number_processes):
		p = Process(target = ImgWorker, args = (dz_queue,))
		#p.deamon = True
		p.setDaemon = True
		p.start()
		procs.append(p)
	'''
    # ファイルのリストをabc順に並び替える
	files = sorted(files)
    # 画像ファイルの数だけ繰り返し
	for imgNb in range(len(files)):
        # filename = 0個目から順に取り出したファイルパス
		filename = files[imgNb]
		#print(filename)
        # opts.basenameJPG = ファイル名(パスなし)の拡張子を除いた部分
		opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
        # 処理中のファイル名と拡張子を表示
		print("processing: " + opts.basenameJPG + " with extension: " + ImgExtension)
		#opts.basenameJPG = os.path.splitext(os.path.basename(slidepath))[0]
		#if os.path.isdir("%s_files" % (basename)):
		#	print("EXISTS")
		#else:
		#	print("Not Found")

        # 画像拡張子に"dcm"が含まれている場合、ファイル名のdcmをjpgに変換する旨を記述
		if ("dcm" in ImgExtension) :
			print("convert %s dcm to jpg" % filename)
            # opts.tmp_dcmオプションがない場合、dcm->jpg変換時の中間ファイル用の出力フォルダ名が無いエラー出力
			if opts.tmp_dcm is None:
				parser.error('Missing output folder for dcm>jpg intermediate files')
            # opts.tmp_dcmがディレクトリでない場合、dcm->jpg変換時の中間ファイル用の出力フォルダ名が無いエラー出力
			elif not os.path.isdir(opts.tmp_dcm):
				parser.error('Missing output folder for dcm>jpg intermediate files')

            # ファイル名の後ろから3文字(拡張子部分)が"jpg"の場合はcontinue
			if filename[-3:] == 'jpg':
                            continue
            # dicomファイルを読み取るコマンドでfilenameファイルを読み込む
            # dicomファイルはwindowsでは通常開くことができないタイプの画像ファイル(レントゲン画像等)
			ImageFile=dicom.read_file(filename)
            # im1 = 画像データ中の濃度情報(numpy配列)
			im1 = ImageFile.pixel_array
            # 最大濃度を取得
			maxVal = float(im1.max())
            # 最小濃度を取得
			minVal = float(im1.min())
            # 画像高さ取得
			height = im1.shape[0]
            # 画像幅取得
			width = im1.shape[1]
            # uint8(画像用の形式[0~255])の高さ×幅×チャンネル数(3)で要素が0のnumpy配列を作成
			image = np.zeros((height,width,3), 'uint8')
            # 濃度情報を小数変換して、最小濃度を引く(小数変換しないと型の影響で引き算が上手くいかないため)
            # 正規化？
			image[...,0] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,1] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,2] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			# dcm_ID = os.path.basename(os.path.dirname(filename))
			# opts.basenameJPG = dcm_ID + "_" + opts.basenameJPG
            # filename＝jpgのファイル名としてtmp_dcm, basenameJPG+"jpg"を結合する
			filename = os.path.join(opts.tmp_dcm, opts.basenameJPG + ".jpg")
			# print(filename)
            # 画像を保存する
			imsave(filename,image)

            # 「opts.basename = スライドのフォルダ名?」, 「opts.basenameJPG = 画像ファイル名」結合してoutputに入れる
			output = os.path.join(opts.basename, opts.basenameJPG)

			try:
                # DeepZoomStaticTilerクラスのrunメソッドを実行
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
			except Exception as e:
				print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
				print(e)

		#elif ("jpg" in ImgExtension) :
		#	output = os.path.join(opts.basename, opts.basenameJPG)
		#	if os.path.exists(output + "_files"):
		#		print("Image %s already tiled" % opts.basenameJPG)
		#		continue

		#	DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()

		elif opts.xmlfile != '':
			xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + '.xml')
			# print("xml:")
			# print(xmldir)
			if os.path.isfile(xmldir):
				if (opts.mask_type==1) or (opts.oLabelref!=''):
					# either mask inside ROI, or mask outside but a reference label exist
					xml_labels, xml_valid = xml_read_labels(xmldir, opts.Fieldxml)
					if (opts.mask_type==1):
						# No inverse mask
						Nbr_ROIs_ForNegLabel = 1
					elif (opts.oLabelref!=''):
						# Inverse mask and a label reference exist
						Nbr_ROIs_ForNegLabel = 0

					for oLabel in xml_labels:
						# print("label is %s and ref is %s" % (oLabel, opts.oLabelref))
						if (opts.oLabelref in oLabel) or (opts.oLabelref==''):
							# is a label is identified 
							if (opts.mask_type==0):
								# Inverse mask and label exist in the image
								Nbr_ROIs_ForNegLabel += 1
								# there is a label, and map is to be inverted
								output = os.path.join(opts.basename, oLabel+'_inv', opts.basenameJPG)
								if not os.path.exists(os.path.join(opts.basename, oLabel+'_inv')):
									os.makedirs(os.path.join(opts.basename, oLabel+'_inv'))
							else:
								Nbr_ROIs_ForNegLabel += 1
								output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
								if not os.path.exists(os.path.join(opts.basename, oLabel)):
									os.makedirs(os.path.join(opts.basename, oLabel))
							if 1:
							#try:
								DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
							#except:
							#	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						if Nbr_ROIs_ForNegLabel==0:
							print("label %s is not in that image; invert everything" % (opts.oLabelref))
							# a label ref was given, and inverse mask is required but no ROI with this label in that map --> take everything
							oLabel = opts.oLabelref
							output = os.path.join(opts.basename, opts.oLabelref+'_inv', opts.basenameJPG)
							if not os.path.exists(os.path.join(opts.basename, oLabel+'_inv')):
								os.makedirs(os.path.join(opts.basename, oLabel+'_inv'))
							if 1:
							#try:
								DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
							#except:
							#	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

				else:
					# Background
					oLabel = "non_selected_regions"
					output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
					if not os.path.exists(os.path.join(opts.basename, oLabel)):
						os.makedirs(os.path.join(opts.basename, oLabel))
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
					except Exception as e:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						print(e)

			else:
				if (ImgExtension == ".jpg") | (ImgExtension == ".dcm") :
					print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
					output = os.path.join(opts.basename, opts.basenameJPG)
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
					except Exception as e:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						print(e)


				else:
					print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (opts.basenameJPG, xmldir) )
					continue
		else:
			output = os.path.join(opts.basename, opts.basenameJPG)
			if os.path.exists(output + "_files"):
				print("Image %s already tiled" % opts.basenameJPG)
				continue
			try:
			#if True:
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml).run()
			except Exception as e:
				print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
				print(e)
	'''
	dz_queue.join()
	for i in range(opts.max_number_processes):
		dz_queue.put( None )
	'''

	print("End")
