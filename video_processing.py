"""Process video."""
import skvideo.io

import os
from PIL import Image

import numpy as np
import data_processing


def extract_frames_of(video_name, ratio_frames):
    """Turn a video into frames."""
    try:
        os.mkdir(str(video_name) + "_frames")
    except:
        return
    reader = skvideo.io.FFmpegReader(video_name)
    frame_count = 0
    for frame in reader.nextFrame():
        if frame_count % int(1 / ratio_frames) == 0:
            im = Image.fromarray(frame)
            im.save(str(video_name) + "_frames/" +
                    str(frame_count) + ".jpeg", "JPEG")

        frame_count += 1
    return


def normalize_video_length(frames_per_sample, frames_names):
    """Normalize the video lengths with 3 rules.

    - if the video is too long ( >60 ), it returns  limit ( integer ) to get
    60 groups of frames in the middle
    - if the video is too short ( < 30 ), it returns integer to copy itself to
    get 20 groups of frames
    - else, it doesnt change
    """
    n_groups_of_frames_wanted_min = 15
    n_groups_of_frames_wanted_max = 25

    n_frames = len(frames_names)
    ratio_frames = n_frames / frames_per_sample
    if ratio_frames >= 60:
        borns = [(len(frames_names) - n_groups_of_frames_wanted_max) / 2,
                 (len(frames_names) + n_groups_of_frames_wanted_max) / 2 - 1]
    elif ratio_frames <= 30:
        borns = [0, n_groups_of_frames_wanted_min]
    elif ratio_frames > 30 and ratio_frames < 60:
        borns = [0, n_groups_of_frames_wanted_min]

    return borns


def get_a_video_as_array(size, prefix, frames_per_sample, video_name):
    """Turn one video into array."""
    video = []
    n_groups_of_frames_per_video = 0

    folder = video_name + "_frames"

    numb_of_frames = len(
        data_processing.get_all_names_from_path(prefix + folder))
    frames_names = []
    for i in range(numb_of_frames):
        frames_names.append(str(i * 5) + ".jpeg")

    group_of_frames = np.zeros((((frames_per_sample, *size, 3))))

    borns = normalize_video_length(frames_per_sample, frames_names)

    for i in range(int(borns[0] * frames_per_sample),
                   int(borns[1] * frames_per_sample)):

        name = frames_names[i % len(frames_names)]

        if i % (frames_per_sample) != 0 or i == 0:
            group_of_frames[frames_names.index(name)
                            % frames_per_sample] = data_processing.jpeg_to_array(
                prefix + folder + "/" + name, size)
        else:
            # The number of frames is too high for the length asked :
            # We save what we have
            video.append(np.array(group_of_frames))
            n_groups_of_frames_per_video += 1
            group_of_frames = np.zeros((((frames_per_sample, *size, 3))))
            group_of_frames[0] = data_processing.jpeg_to_array(
                prefix + folder + "/" + name, size)

    # We add the last one
    video.append(group_of_frames)
    n_groups_of_frames_per_video += 1

    return video, n_groups_of_frames_per_video


def main(name_data_set, ratio_frames):
    """Launch the video processing."""
    if name_data_set == "Windstorm":
        video_path = data_processing.get_path("Videos", "Videos")
        video_names = data_processing.get_all_names_from_path(video_path)
        for video_name in video_names:

            extract_frames_of(video_path + video_name, ratio_frames)
            print("frames of : {0} extracted".format(str(video_name)))


if __name__ == '__main__':
    ratio_frames = 1 / 5
    name_data_set = "Windstorm"
    main(name_data_set, ratio_frames)
