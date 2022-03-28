from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.DATASET_ABS = 'Phys'
_C.PHYRE_PROTOCAL = 'cross'
_C.PHYRE_FOLD = 9
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode()
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.LR_GAMMA = 0.1

_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.SCHEDULER = 'step'
# prediction setting
_C.PRED_SIZE_TRAIN = 20
_C.PRED_SIZE_TEST = 40

_C.HORIZONTAL_FLIP = False
_C.VERTICAL_FLIP = False
# input for mixed dataset
_C.IMAGE_EXT = '.jpg'
_C.INPUT_HEIGHT = 224  # 128
_C.INPUT_WIDTH = 224  # 128



