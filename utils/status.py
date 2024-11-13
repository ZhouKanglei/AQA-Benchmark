#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/10 15:59:54


import sys
from datetime import datetime
from time import time
from typing import Union


class ProgressBar:
    def __init__(self, verbose=True):
        self.old_time = 0
        self.running_sum = 0
        self.verbose = verbose

    def prog(self, i: int, max_iter: int, epoch: Union[int, str], loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.
        :param i: the current iteration
        :param max_iter: the maximum number of iteration
        :param epoch: the epoch
        :param loss: the current value of the loss function
        """
        if not self.verbose:
            if i == 0:
                print(
                    "\r{} | Epoch {:03d}\n".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch + 1,
                    ),
                    file=sys.stderr,
                    end="",
                    flush=True,
                )
            else:
                return
        if i == 0:
            self.old_time = time()
            self.running_sum = 0
        else:
            self.running_sum = self.running_sum + (time() - self.old_time)
            self.old_time = time()
        if i:  # not (i + 1) % 10 or (i + 1) == max_iter:
            progress = min(float((i + 1) / max_iter), 1)
            progress_bar = ("█" * int(36 * progress)) + (
                "┈" * (36 - int(36 * progress))
            )
            print(
                "\r{} + Epoch {:03d}: |{}| {} ep/h | loss: {:.2f} |".format(
                    datetime.now().strftime("%y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    progress_bar,
                    round(3600 / (self.running_sum / i * max_iter), 2),
                    round(loss, 8),
                ),
                file=sys.stderr,
                end="" if i + 1 < max_iter else "\n",
                flush=True,
            )


def progress_bar(i: int, max_iter: int, epoch: Union[int, str], loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param loss: the current value of the loss function
    """
    global static_bar

    if i == 0:
        static_bar = ProgressBar()
    static_bar.prog(i, max_iter, epoch, loss)
