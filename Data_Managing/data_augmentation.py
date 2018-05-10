import numpy as np


def cut_long_videos(names, labels, videos, threshold):
    print('format video pre cut ', np.array(videos).shape)
    videos_cutted, names_cutted, labels_cutted = [], [], []
    for video, name, label in zip(videos, names, labels):
        if len(video) > threshold:
            n_cutted = int(len(video)/threshold)+1
            step = len(video)%n_cutted
            videos_cutted.extend(np.array([video[i * step:i * step + threshold] for i in range(n_cutted)]))
            names_cutted.extend([name] * n_cutted)
            labels_cutted.extend([label] * n_cutted)

    return np.array(videos_cutted), np.array(names_cutted), np.array(labels_cutted)
