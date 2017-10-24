"""Launchs the youtube bot."""

import youtubewithpytube
from bs4 import BeautifulSoup
from urllib.request import urlopen


def get_video_downloaded():
    """Get video names allready downloaded."""
    with open("videos_already_downloaded.txt") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        return content


def write_video_downloaded(video):
    """Write video name into the text file."""
    with open("videos_already_downloaded.txt", "a") as f:
        f.write(video + "\n")
    f.close()


def get_urls_from_yt_page(url_name):
    """Get all URL from the html code of the webpage, given an url name."""
    redditFile = urlopen(url_name)
    redditHtml = redditFile.read()
    redditFile.close()

    soup = BeautifulSoup(redditHtml, "lxml")

    listOfLinks = []
    for links in soup.find_all('a'):
        a = links.get('href')
        # All videos woth https in the name refer to the initial video
        # and we are searching for suggested videos
        if "http" not in str(a) and "/watch" in str(a):
            a = "https://www.youtube.com" + a  # To make a real link
            listOfLinks.append(a)
    return listOfLinks


def main(path):
    """Launch the bot."""
    linkToBeginWith = "https://www.youtube.com/watch?v=Z25-kHlgz44"
    URLlistToexplore = [linkToBeginWith]
    videos_downloaded = get_video_downloaded()

    numberVideosWanted = 100

    while len(videos_downloaded) - numberVideosWanted != 0:
        print("URLs to explore on this iteration : {0}".format(
            URLlistToexplore))
        for url in URLlistToexplore:
            url_not_seen = [url for url in get_urls_from_yt_page(url)
                            if url not in videos_downloaded]
            URLlistToexplore = URLlistToexplore + url_not_seen
            write_video_downloaded(url)

            youtubewithpytube.dl_yt_video(url, path)
            videos_downloaded.append(url)
            print("video downloaded : {}".format(url))


if __name__ == '__main__':
    path = 'Videos'
    main(path)
