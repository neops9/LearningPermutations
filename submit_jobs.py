#!/usr/bin/python3
# coding: utf-8

import shlex
from importlib.machinery import SourceFileLoader
import math
import itertools
import copy
import subprocess
import os
import datetime
import multiprocessing
import sys
import re
import argparse


# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default=os.getcwd(), help="Destination path")
parser.add_argument("--train", dest="parser_path", type=str, required=True, help="train program to run")
parser.add_argument("--gres", type=str, default="gpu:1", help="Ressources to allocate")
parser.add_argument("--mem", type=str, default="", help="Ressources to allocate")
parser.add_argument("--submit", action="store_true", help="Submit jobs with sbatch")
parser.add_argument('file', metavar='FILE', type=str, help='Configuration file')
args = parser.parse_args()


# format a dict as command line arguments
def to_args(d):
    ret = ""
    for k, v in d.items():
        ret += " " + str(k) + " " + str(v)
    return ret


# perform grid search
def grid_search(settings, options):
    for l in itertools.product(*options):
        names = []
        settings2 = copy.deepcopy(settings)
        for name, v in l:
            settings2.update(v)
            names.append(name)
        yield names, settings2


def launch(launch_args, gres, mem, submit=False):
    name, dir_path, custom_cmd = launch_args

    # create output dir
    odir = os.path.join(dir_path, str(name))
    os.mkdir(odir)

    # train command
    custom_cmd.update(
        {
            "--model": os.path.join(odir, "model"),
            "--tensorboard": os.path.join(os.path.join(dir_path, "tb", str(name)))
        }
    )

    # launch
    command = "python " + args.parser_path
    command += to_args(custom_cmd)
    command_path = os.path.join(odir, "cmd")
    with open(command_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n"%name)
        f.write("#SBATCH --output=%s\n"%os.path.join(odir, "out"))
        f.write("#SBATCH --error=%s\n"%os.path.join(odir, "err"))
        f.write("#SBATCH --gres=%s\n" % gres)
        f.write("#SBATCH --time=10:00:00\n")
        f.write("#SBATCH --cpus-per-task=3\n")
        f.write("#SBATCH --ntasks=1\n")
        #f.write("#SBATCH --hint=nomultithread\n")
        if len(mem) > 0:
            f.write("#SBATCH --mem %s\n" % mem)
        f.write(command)
        f.write("\n")

    # launch
    print("Job created: %s"%name)
    if submit:
        subprocess.Popen(["sbatch", "cmd"], shell=False, cwd=odir).wait()


# read configuration
config = SourceFileLoader('config', args.file).load_module()

# command line options
cmd = {}
cmd_options = []
if hasattr(config, 'cmd'):
    cmd = config.cmd
if hasattr(config, 'cmd_options'):
    cmd_options = config.cmd_options

# default values if not provided
if len(cmd_options) == 0:
    cmd_options = [[('default', {})]]


dir_path = os.path.join(args.output, str(datetime.datetime.now()).replace(" ", "_"))
os.mkdir(dir_path)
count = 0
proc_args = []
for names1, custom_cmd in grid_search(cmd, cmd_options):
    count = count + 1
    name = str(count) + "[" + ",".join(names1) + "]"
    launch_args = (name, dir_path, custom_cmd)
    proc_args.append(launch_args)

# create jobs
for launch_args in proc_args:
    launch(launch_args, args.gres, args.mem, args.submit)
