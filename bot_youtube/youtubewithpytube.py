"""Downloads a youtube video given a link."""

from pytube import YouTube


def dl_yt_video(link, repertory):
    """Dl a youtube video given a link."""
    YouTube(link).streams.first().download(repertory)


if __name__ == '__main__':
    while True:
        link = input("please enter a link")
        dl_yt_video(link, "videos")
