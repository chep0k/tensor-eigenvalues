import matplotlib.pyplot as plt
import typing as tp


def plot_4(x1: list[float], y1: list[float],
           x2: list[float], y2: list[float],
           x3: list[float], y3: list[float],
           x4: list[float], y4: list[float],
           titles: list[str], suptitle: str,
           figsize=(12, 8), supt_fontsize=18):
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(suptitle, fontsize=supt_fontsize)
    fig.subplots_adjust(top=0.9)

    pass
