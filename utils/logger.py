#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 15:38:34


import logging


class Logger(object):
    """
    Logger for logging.
    """

    def __init__(self, log_file, exp_name):
        """
        Init logger.
        """
        self.log_file = log_file
        self.exp_name = exp_name

        # initialize with args
        self.init()

    def init(self):
        """
        Initialize.
        """
        # logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_fh = logging.FileHandler(self.log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        # set logger
        self.logger = logger.info

    def info(self, log=None, level=-1):
        """
        Info log.
        """
        if log is None:
            log = ""
            self.logger(log.center(78, "-"))

        if level == -1:
            # log = log.replace(self.exp_name, "{exp name}")
            self.logger(log)

        if level == 0:
            self.logger("".center(78, "-"))
            log = "| " + log + " |"
            self.logger(log.center(78, "-"))
            self.logger("".center(78, "-"))

        if level == 1:
            length = len(log)

            self.logger(log + " " + "." * (78 - length - 1))

        if level == 2:
            self.logger((" " + log + " ").center(78, "*"))
