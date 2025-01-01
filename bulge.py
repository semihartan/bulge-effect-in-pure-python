import os
import cv2
import math
import numpy as np
from enum import IntEnum
import threading

class BulgeQuality(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    HIGHEST = 3

class BulgeOptions:
    def __init__(self, x, y, radius, scale=-0.02, amount=1.5, quality=BulgeQuality.NORMAL) -> None:
        self.center = np.array([x, y], np.float32)
        self.radius = radius
        self.scale = scale
        self.amount = amount
        self.quality = quality

class BulgeEffect:
    INTERP_TYPES = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    def __init__(self, img, options=None): 
        self._map_x = None
        self._map_y = None
        self._img = img
        self._img_original = img.copy()
        self._previous_indices = None
        if options is None:
            options = BulgeOptions(50, 50, 50)
        self._options = options
        self._worker_count = 1
        self._divisors = self._closest_divisors(self._worker_count)
        # os.environ['TF_CONFIG'] = json.dump(tf_config)
    
    @property
    def amount(self):
        return self._options.amount

    @property
    def center(self):
        return self._options.center
    
    @property
    def image(self):
        return self._img
    
    @property
    def quality(self):
        return self._options.quality
    
    @property
    def radius(self):
        return self._options.radius
    
    @property
    def scale(self):
        return self._options.scale

    @image.setter
    def image(self, image):
        self._img = image

    def set_options(self, options):
        self._options = options

    def apply(self):
        if self._previous_indices:
            py0 = self._previous_indices[0] 
            py1 = self._previous_indices[1]
            px0 = self._previous_indices[2]
            px1 = self._previous_indices[3]

            self._img[py0:py1, px0:px1] = self._img_original[py0:py1, px0:px1] 
        threads = []
        for i in range(self._worker_count):
            thread = threading.Thread(target=BulgeEffect._apply_core, args=((self, i, self._divisors[0], self._divisors[1]),))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        H = self._img.shape[0]
        W = self._img.shape[1]
        x0 = np.clip(self._options.center[0] - self._options.radius, 0, W).astype(dtype=np.int32)
        x1 = np.clip(self._options.center[0] + self._options.radius, 0, W).astype(dtype=np.int32)
        y0 = np.clip(self._options.center[1] - self._options.radius, 0, H).astype(dtype=np.int32)
        y1 = np.clip(self._options.center[1] + self._options.radius, 0, H).astype(dtype=np.int32)
         
        self._previous_indices = (y0, y1, x0, x1)

    def _closest_divisors(self, n):
        a = round(math.sqrt(n))
        while n%a > 0: a -= 1
        return a,n//a

    @staticmethod
    def _apply_core(args):
        self = args[0]
        k = args[1]
        row_count = args[2]
        column_count = args[3]
        
        W = img.shape[1]
        H = img.shape[0]

        center = np.array(self._options.center, dtype=np.int32)
        radius = np.array(self._options.radius, dtype=np.int32)

        block_sz_w = 2 * radius / column_count 
        block_sz_h = 2 * radius / row_count

        i = k // column_count
        j = k % column_count

        f0x = center[0] - radius
        f0y = center[1] - radius
        
        x0 = np.clip(f0x + j * block_sz_w, 0, W).astype(dtype=np.int32)
        x1 = np.clip(f0x + (j + 1) * block_sz_w, 0, W).astype(dtype=np.int32)
        y0 = np.clip(f0y + i * block_sz_h, 0, H).astype(dtype=np.int32)
        y1 = np.clip(f0y + (i + 1) * block_sz_h, 0, H).astype(dtype=np.int32)
        
        center = center.astype(dtype=np.float32)
        radius = radius.astype(dtype=np.float32)

        radius2 = np.power(radius, 2)
        _map_y, _map_x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        
        dv_x = _map_x - center[0]
        dv_y = _map_y - center[1]

        distance2 = dv_x ** 2 + dv_y ** 2
        # circle_indices = np.where(distance2 <= radius2)
        circle_indices = distance2 <= radius2
        
        radius_normalized = distance2 / radius2
        # Our translate function is s * e^(-0.5 * (x^2) / (a^2)), this function requires
        # a x value large enough to refer to the border of the circle. So multiply
        # the x value by 16, which is large enough
        radius_normalized = radius_normalized * 16
        radius_normalized = self._options.scale * np.exp(-0.5 * (radius_normalized / self._options.amount) ** 2)

        _map_y[circle_indices] = _map_y[circle_indices] - dv_y[circle_indices] * radius_normalized[circle_indices]
        _map_x[circle_indices] = _map_x[circle_indices] - dv_x[circle_indices] * radius_normalized[circle_indices]
        
        _img_mapped = cv2.remap(self._img, _map_x, _map_y, 
            self._get_interp_from_quality(self.quality), borderMode=cv2.BORDER_REPLICATE)
        self._img[y0:y1, x0:x1] = _img_mapped
 
    def _get_interp_from_quality(self, quality):
        return BulgeEffect.INTERP_TYPES[quality]
    
    def _determine_worker_count(self):
        logical_core_count = os.cpu_count()
        if logical_core_count < 4:
            return 1
        worker_count = 4
        # Assumes core count is multiple of 2
        while worker_count < logical_core_count:
            worker_count *= worker_count
        return worker_count

def on_mouse_move(x, y, button):
    if img is None:
        return
    
    bulge_effect.set_options(BulgeOptions(x, y, 200, scale=0.9, amount=1.4, quality=BulgeQuality.NORMAL))
    bulge_effect.apply() 

    cv2.imshow("buldge", bulge_effect.image) 

def get_button(flags):
    if flags & cv2.EVENT_FLAG_LBUTTON:
        return 'left'
    if flags & cv2.EVENT_FLAG_RBUTTON:
        return 'right'
    if flags & cv2.EVENT_FLAG_MBUTTON:
        return 'middle'
    return 'none'

def mouse_event_handler(event, x, y, flags, param):
    button_down = [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]
    button_up = [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP]
    button = get_button(flags) 
    if event == cv2.EVENT_MOUSEMOVE:
        on_mouse_move(x, y, button)


if __name__ == "__main__":
    global img, bulge_effect
    img = cv2.imread('test6.png')
    _map_y, _map_x = np.mgrid[0:10, 0:5].astype(np.float32)
    
    bulge_effect = BulgeEffect(img)
    bulge_effect.set_options(BulgeOptions(50, 50, 50, scale=-0.4, amount=1.4, quality=BulgeQuality.NORMAL))
    bulge_effect.apply() 
    cv2.imshow('buldge', img)
    cv2.setMouseCallback('buldge', mouse_event_handler)
    
    cv2.waitKey(0)