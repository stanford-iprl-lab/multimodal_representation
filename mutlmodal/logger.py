import git
from tensorboardX import SummaryWriter
import datetime
import time
import os

import logging
import sys
import yaml


class Logger(object):
    """
    Hooks for print statements and tensorboard logging
    """

    def __init__(self, configs):

        self.configs = configs

        time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M")
        prefix_str = time_str + "_" + configs["notes"]
        if configs["dev"]:
            prefix_str = "dev_" + prefix_str

        self.log_folder = os.path.join(self.configs["logging_folder"], prefix_str)
        self.tb_prefix = prefix_str

        self.setup_checks()
        self.create_folder_structure()
        self.setup_loggers()
        self.dump_init_info()

    def create_folder_structure(self):
        """
        Creates the folder structure for logging. Subfolders can be added here
        """
        base_dir = self.log_folder
        sub_folders = ["runs", "models"]

        if not os.path.exists(self.configs["logging_folder"]):
            os.mkdir(self.configs["logging_folder"])

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        for sf in sub_folders:
            if not os.path.exists(os.path.join(base_dir, sf)):
                os.mkdir(os.path.join(base_dir, sf))

    def setup_loggers(self):
        """
        Sets up a logger that logs to both file and stdout
        """
        log_path = os.path.join(self.log_folder, "log.log")

        self.print_logger = logging.getLogger()
        self.print_logger.setLevel(
            getattr(logging, self.configs["log_level"].upper(), None)
        )
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        for h in handlers:
            h.setFormatter(formatter)
            self.print_logger.addHandler(h)

        # Setup Tensorboard
        self.tb = SummaryWriter(os.path.join(self.log_folder, "runs", self.tb_prefix))

    def setup_checks(self):
        """
        Verifies that all changes have been committed
        Verifies that hashes match (if continuation)
        """
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha


        # Test for continuation
        if self.configs["continuation"]:
            self.log_folder = self.configs["logging_folder"]
            with open(os.path.join(self.log_folder, "log.log"), "r") as old_log:
                for line in old_log:
                    find_str = "Git hash"
                    if line.find(find_str) is not -1:
                        old_sha = line[line.find(find_str) + len(find_str) + 2 : -4]
                        assert sha == old_sha

    def dump_init_info(self):
        """
        Saves important info for replicability
        """
        if not self.configs["continuation"]:
            self.configs["logging_folder"] = self.log_folder
        else:
            self.print("=" * 80)
            self.print("Continuing log")
            self.print("=" * 80)

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        self.print("Git hash: {}".format(sha))
        self.print("Dumping YAML file")
        self.print("Configs: ", yaml.dump(self.configs))

        # Save the start of every run
        if "start_weights" not in self.configs:
            self.configs["start_weights"] = []
        self.configs["start_weights"].append(self.configs["load"])

        with open(os.path.join(self.log_folder, "configs.yml"), "w") as outfile:
            yaml.dump(self.configs, outfile)
            self.tb.add_text("hyperparams", str(self.configs))

    def end_itr(self, weights_path):
        """
        Perform all operations needed at end of iteration
        1). Save configs with latest weights
        """
        self.configs["latest_weights"] = weights_path
        with open(os.path.join(self.log_folder, "configs.yml"), "w") as outfile:
            yaml.dump(self.configs, outfile)

    def print(self, *args):
        """
        Wrapper for print statement
        """
        self.print_logger.info(args)

