# utils.py

import argparse 


def open_label_file(path):

	with open(path, 'r') as f:
		labels = [int(l) for l in f.read().splitlines()]

	return labels 


