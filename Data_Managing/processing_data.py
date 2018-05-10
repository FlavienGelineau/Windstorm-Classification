from PIL import Image
import numpy as np

def process_frame_rate(video, frame_rate, frame_rate_wanted):
    multiplier= round(frame_rate/frame_rate_wanted)
    i_max = int(len(video)/multiplier)
    return np.array([video[i*multiplier] for i in range(i_max)])

def process_frame_shape(video, frame_shape_wanted):
    def reshape_array(img_array):
        img = Image.fromarray(img_array)
        img = img.resize(frame_shape_wanted, Image.ANTIALIAS)
        return np.asarray(img, dtype="int32")

    return np.array([reshape_array(frame) for frame in video])