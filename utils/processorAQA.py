#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 16:53:32


from utils.processor import Processor, torch
from utils.status import ProgressBar
from utils.evaluator import Evaluator


class ProcessorAQA(Processor):
    """
    Processor for AQA.
    """

    def __init__(self, args):
        """
        Init processor.
        """
        super(ProcessorAQA, self).__init__(args)

    def train(self, start_epoch=0):
        """
        Train.
        """
        self.logger.info(f"Training - {self.exp_name}", level=0)
        self.logger.info("Start training: ", level=1)

        self.model.train()

        progress_bar = ProgressBar()
        evaluator = Evaluator(self.log_dir, "train")
        val_evaluator = Evaluator(self.log_dir, "val")

        for epoch in range(start_epoch, self.args.epoch):
            for i, batch_data in enumerate(self.train_loader):

                batch_data = self.data2device(batch_data)
                # training loop
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs)
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                evaluator.add_data(outputs)
                progress_bar.prog(i, len(self.train_loader), epoch, loss.item())

            # calculate metrics for training set
            evaluator.calculate_metrics()
            self.logger.info(
                f"Epoch {epoch+1:03d}, Train ({evaluator.num_samples:4d}): "
                f"SRCC = {evaluator.srcc:.4f}, "
                f"L2 = {evaluator.l2:8.4f}, "
                f"RL2 = {evaluator.rl2e:.4f}",
                level=1,
            )
            # calculate metrics for validation set
            self.eval(val_evaluator, epoch)
            # find the best model
            if val_evaluator.best_srcc_epoch == epoch:
                self.logger.info(
                    f"Best SRCC: {val_evaluator.best_srcc:.4f}"
                    f" at epoch {val_evaluator.best_srcc_epoch + 1:03d}",
                    level=2,
                )
                save_dict = {
                    "epoch": epoch,
                    "best_srcc": val_evaluator.best_srcc,
                    "best_srcc_epoch": val_evaluator.best_srcc_epoch,
                    "L2": val_evaluator.l2,
                    "RL2": val_evaluator.rl2e,
                }
                self.save_model_weight(dict=save_dict)

        self.logger.info("Finish training.", level=1)

    def eval(self, evaluator, epoch):
        """
        Evaluate.
        """

        status = self.model.training
        self.model.eval()

        for i, batch_data in enumerate(self.test_loader):

            batch_data = self.data2device(batch_data)

            with torch.no_grad():
                outputs = self.model(batch_data)

            evaluator.add_data(outputs)

        evaluator.calculate_metrics()
        self.logger.info(
            f"Epoch {epoch+1:03d},  Eval ({evaluator.num_samples:4d}): "
            f"SRCC = {evaluator.srcc:.4f}, "
            f"L2 = {evaluator.l2:8.4f}, "
            f"RL2 = {evaluator.rl2e:.4f}"
        )

        self.model.train(status)

    def test(self):
        """
        Test.
        """
        self.logger.info(f"Testing - {self.exp_name}", level=0)
        self.logger.info("Start testing: ", level=1)

        status = self.model.training
        self.model.eval()

        progress_bar = ProgressBar()
        evaluator = Evaluator(self.log_dir, "test")

        for i, batch_data in enumerate(self.test_loader):
            batch_data = self.data2device(batch_data)

            with torch.no_grad():
                outputs = self.model(batch_data)

            evaluator.add_data(outputs)
            progress_bar.prog(i, len(self.test_loader), 0, 0)

        evaluator.calculate_metrics()
        self.logger.info(
            f"Test ({evaluator.num_samples:4d}): "
            f"SRCC = {evaluator.srcc:.4f}, "
            f"L2 = {evaluator.l2:8.4f}, "
            f"RL2 = {evaluator.rl2e:.4f}",
            level=1,
        )
        self.save_best_result(evaluator)
        # measure the performance and save the result
        self.measure_model(batch_data)

        self.logger.info("Finish testing.", level=1)
        self.model.train(status)

    def process(self):
        """
        Process.
        """
        if self.args.phase == "train":
            epoch = (
                self.epoch if self.args.continue_train and hasattr(self, "epoch") else 0
            )
            if self.args.continue_train:
                self.load_model_weight(weight_path=self.args.weight_path)
            self.train(start_epoch=epoch)
        # load the best model and test
        self.load_model_weight(weight_path=self.args.weight_path)
        self.test()
