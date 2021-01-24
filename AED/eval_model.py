from utils import evals
import argparse

"""
Usage: python eval_model.py gtlabels.csv estlabels.csv 2
"""

parser = argparse.ArgumentParser(description="Task ID")
parser.add_argument("gt", type=str, help="ground truth labels")
parser.add_argument("est", type=str, help="estimated labels")
parser.add_argument("taskid", type=int, help="task id {1,2}")
args = parser.parse_args()

assert args.taskid in [1, 2], args.taskid
score = evals(args.gt, args.est, args.taskid)
print("Your score is: %d/100" % (score * 100))
