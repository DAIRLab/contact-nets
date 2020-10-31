import os.path as op

ROOT_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
def root_path(*path):
    return op.join(ROOT_DIR, *path)

OUT_DIR = root_path("out")
def out_path(*path):
    return op.join(OUT_DIR, *path)

DATA_DIR = root_path("data")
def data_path(*path):
    return op.join(DATA_DIR, *path)

RESULTS_DIR = root_path("results")
def results_path(*path):
    return op.join(RESULTS_DIR, *path)

LIB_DIR = root_path("lib")
def lib_path(*path):
    return op.join(LIB_DIR, *path)

PROCESSING_DIR = root_path("contactnets", "utils", "processing")
def processing_path(*path):
    return op.join(PROCESSING_DIR, *path)
