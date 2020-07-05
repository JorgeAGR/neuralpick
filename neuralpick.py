# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:10:56 2020

@author: jorge
"""

import os
import argparse
from src.models import PickingModel, CheckingModel
from src.aux_funcs import read_Config
from src.phase_picker import Picker
from src.scanner import Scanner

def pick(model_type, args):
    file_dir = args.file_path
    if file_dir[-1] != '/':
        file_dir += '/'
    if model_type == 'p':
        model = PickingModel(args.model_name)
        Picker(file_dir, model, args.phase, overwrite=args.nooverwrite)
    else:
        print('ERROR: Invalid model type. Use a picking model.')
    return

def check(model_type, args):
    file_dir = args.file_path
    if file_dir[-1] != '/':
        file_dir += '/'
    if model_type == 'c':
        model = CheckingModel(args.model_name)
        print('Coming soon!')
    else:
        print('ERROR: Invalid model type. Use a checking model.')
    return

def scan(model_type, args):
    file_dir = args.file_path
    if file_dir[-1] != '/':
        file_dir += '/'
    if model_type == 'p':
        model = PickingModel(args.model_name)
        Scanner(file_dir, model, args.phase, args.begin, args.end, args.number)
    else:
        print('ERROR: Invalid model type. Use a picking model.')
    return

def train(model_type, args):
    if model_type == 'p':
        model = PickingModel(args.model_name)
    elif model_type == 'c':
        model = CheckingModel(args.model_name)
    else:
        print('ERROR: Invalid model type. Assign a valid model type.')
        return
    if args.force:
        model.trained = False
    model.train_Model()
    model.save_Model()
    if model.trained:
        print('Model already exists. Use -f option to force model training and overwrite previous model.')
    return

parser = argparse.ArgumentParser(description='Software for training and deploying CNNs for seismic data quality checking and phase identification.')

subparsers = parser.add_subparsers()

parser_pick = subparsers.add_parser('pick',
                                    help='Pick the main arrival of a seismic phase in the file header of seismic data using a trained CNN model.')
parser_pick.add_argument('file_path', help='Path to files to pick for.', type=str)
parser_pick.add_argument('phase', help='Seismic phase to pick for (case sensitive).', type=str)
parser_pick.add_argument('model_name', help='Name of the model (barring the .conf extension).', type=str)
parser_pick.add_argument('-no', '--nooverwrite',
                         help='Optional argument to prevent program from overwriting the input SAC files.',
                         action='store_false')
parser_pick.set_defaults(func=pick)

parser_check = subparsers.add_parser('check', help='Quality check seismograms using a CNN model.')
parser_check.add_argument('file_path', help='Path to files to quality check.', type=str)
parser_check.add_argument('phase', help='Seismic phase to quality check around (case sensitive).', type=str)
parser_check.add_argument('model_name', help='Name of the model (barring the .conf extension).', type=str)
parser_check.set_defaults(func=check)

parser_scan = subparsers.add_parser('scan', help='Scan a set time range from a seismic phase to find pre/postcursors.')
parser_scan.add_argument('file_path', help='Path to files to quality check.', type=str)
parser_scan.add_argument('phase', help='Seismic phase to quality check around (case sensitive).', type=str)
parser_scan.add_argument('begin', help='Start time from the main arrival in seconds.', type=float)
parser_scan.add_argument('end', help='End time from the main arrival in seconds.', type=float)
parser_scan.add_argument('model_name', help='Name of the model (barring the .conf extension).', type=str)
parser_scan.add_argument('-n' , '--number', help='Number of relevant predictions to consider', type=float, default=10)
parser_scan.set_defaults(func=scan)

parser_train = subparsers.add_parser('train', help='Train a new picking or checking model using a seismic dataset.')
parser_train.add_argument('model_name', help='Name of the model (barring the .conf extension).', type=str)
parser_train.add_argument('-f', '--force',
                          help='Optional argument to force the training and overwritting of an existing model.',
                          action='store_true')
parser_train.set_defaults(func=train)

args = parser.parse_args()

config = read_Config('models/conf/{}.conf'.format(args.model_name))
model_type = config['model_type'].lower()

args.func(model_type, args)