import numpy as np


def cut_long_videos(names, labels, videos, threshold):
    videos_cutted, names_cutted, labels_cutted = [], [], []
    for video, name, label in zip(videos, names, labels):
        if len(video) > threshold:
            n_cutted = int(len(video)/threshold)
            step = len(video)%n_cutted
            videos_cutted.extend([video[i * step:i * step + threshold] for i in range(n_cutted)])
            names_cutted.extend([name] * n_cutted)
            labels_cutted.extend([label] * n_cutted)
    print("shape videos cutted", np.array(videos_cutted).shape)

    return np.array(videos_cutted), np.array(names_cutted), np.array(labels_cutted)
