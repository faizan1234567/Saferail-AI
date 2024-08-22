import os
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart


# implementation of calibration script
#  # trt.IInt8Calibrator
# calibrator
#IInt8EntropyCalibrator2
#IInt8LegacyCalibrator
#IInt8EntropyCalibrator
#IInt8MinMaxCalibrator
class Calibrator(trt.IInt8MinMaxCalibrator):
    
    def __init__(self, calibaration_path, nCalibration, inputShape, cacheFile):
        trt.IInt8MinMaxCalibrator.__init__(self)

        # self.imageList = glob(calibaration_path + "*.jpg")[:100]
        self.vi = os.path.join(calibaration_path, "vi")
        self.ir = os.path.join(calibaration_path, "ir")
        self.imageList = os.listdir(self.vi)
        self.nCalibration = nCalibration
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        # print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)    
        
    def batchGenerator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))
    
    
    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            vi_img = cv2.imread(os.path.join(self.vi, imageList[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            ir_img = cv2.imread(os.path.join(self.ir, imageList[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            #resize
            vi_img = cv2.resize(vi_img, self.shape[2:])
            ir_img = cv2.resize(ir_img, self.shape[2:])
            
            # expand for batch and channel dim
            vi_img = np.expand_dims(vi_img, axis=[0, 1])
            ir_img = np.expand_dims(ir_img, axis=[0, 1])
            img  = np.concatenate((ir_img, vi_img), axis=1)
            res[i] = img[0]
        return res
    
    def get_batch_size(self):  # necessary API
        return self.shape[0]
    
    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
                return None
    
    def read_calibration_cache(self):  # necessary API
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return
        
    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return


# run everything now
if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = Calibrator("/home/jetson/Faizan/fusion/Saferail-AI/images", 5, (1, 2, 640, 640), "./fusion-int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")