#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/07 21:23:02

from utils.parser import ParseArgs
from utils.processorAQA import ProcessorAQA

if __name__ == "__main__":
    args = ParseArgs().args
    processor = ProcessorAQA(args)
    processor.process()
