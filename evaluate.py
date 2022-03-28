import os
import cv2
import torch
import phyre
import hickle
import random
import numpy as np
from tqdm import tqdm
import timm
from torch import nn
from vit_pytorch import ViT
import argparse
from utils.config import _C as cfg


def arg_parse():
    parser = argparse.ArgumentParser(description='VIT_PHYRE Parameters')
    parser.add_argument('--model', required=True, help='path to tset model', type=str)
    parser.add_argument('--protocal', type=str, help='within or cross', default='within')
    parser.add_argument('--fold', type=int, help='from 0 to 9', default=0)
    parser.add_argument('--batch_size', type=int, help='test batch size', default=128)

    return parser.parse_args()


args = arg_parse()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("Loading model...")
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(768, 2)
model.load_state_dict(torch.load(f"./models/{args.model}"))
model.to(device)

# model = ViT(
#     image_size = 128,
#     patch_size = 16,
#     num_classes = 1,
#     dim = 512,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512
# ).to(device)

start_id, end_id = 0, 25
random.seed(0)
np.random.seed(0)
# protocal, fold_id = C.PHYRE_PROTOCAL, C.PHYRE_FOLD
protocal, fold_id = args.protocal, args.fold
print(f'testing using protocal {protocal} and fold {fold_id}')

# setup the PHYRE evaluation split
eval_setup = f'ball_{protocal}_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
_, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
simulator = phyre.initialize_simulator(test_list, action_tier)

# the action candidates are provided by the author of PHYRE benchmark
num_actions = 10000
cache = phyre.get_default_100k_cache('ball')
acts = cache.action_array[:num_actions]
training_data = cache.get_sample(test_list, None)

# some statistics variable when doing the evaluation
auccess = np.zeros((len(test_list), 100))
# batched_pred = C.SOLVER.BATCH_SIZE * 10
batched_pred = args.batch_size

all_data = []

with torch.no_grad():
    t_list = tqdm(test_list, 'Task')
    for task_id, task in enumerate(t_list):
        sim_statuses = training_data['simulation_statuses'][task_id]
        confs, successes = [], []

        # act_list = tqdm(acts, 'Candidate Action', leave=False)
        for act_id, act in enumerate(acts):
            sim = simulator.simulate_action(task_id, act, stride=60, need_images=True, need_featurized_objects=True)
            # assert sim.status == sim_statuses[act_id], 'sanity check not passed'
            if sim.status == phyre.SimulationStatus.INVALID_INPUT:
                if act_id == len(acts) - 1 and len(all_data) > 0:  # final action is invalid
                    # raise ValueError("invalid act")
                    out = model(torch.cat(all_data).to(device))
                    out = nn.Softmax(1)(out)
                    conf_t = out[:, 1]
                    confs = confs + conf_t.cpu().detach().numpy().tolist()
                    all_data = []
                continue
            successes.append(sim.status == phyre.SimulationStatus.SOLVED)
            image = cv2.resize(sim.images[0], (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT),
                               interpolation=cv2.INTER_NEAREST)

            image = phyre.observations_to_float_rgb(image)
            data = image.transpose((2, 0, 1))[None, None, :]
            data = torch.from_numpy(data.astype(np.float32))
            all_data.append(data[0])

            if len(all_data) % batched_pred == 0 or act_id == len(acts) - 1:
                out = model(torch.cat(all_data).to(device))
                out = nn.Softmax(1)(out)
                conf_t = out[:, 1]
                confs = confs + conf_t.cpu().detach().numpy().tolist()
                all_data = []

        info = f'current AUCCESS: '
        top_acc = np.array(successes)[np.argsort(confs)[::-1]]
        for i in range(100):
            auccess[task_id, i] = int(np.sum(top_acc[:i + 1]) > 0)
            # 0/1 represents for whether task_id has successed with i attempts
        w = np.array([np.log(k + 1) - np.log(k) for k in range(1, 101)])
        s = auccess[:task_id + 1].sum(0) / auccess[:task_id + 1].shape[0]
        info += f'{np.sum(w * s) / np.sum(w) * 100:.2f}'
        t_list.set_description(info)