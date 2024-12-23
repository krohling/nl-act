from ee_sim_env import (
    GraspCubeEETask,
    StackCubeEETask,
    TransferCubeEETask
)
from sim_env import (
    GraspCubeTask,
    StackCubeTask,
    TransferCubeTask
)
from scripted_policy import (
    GraspCubePolicy,
    StackCubePolicy,
    TransferCubePolicy
)

DEFAULT_EPISODE_LEN = 500

TASK_ID_GRASP_CUBE = 0
TASK_ID_STACK_CUBE = 1
TASK_ID_TRANSFER_CUBE = 2

ALL_TASKS = [
    TASK_ID_GRASP_CUBE,
    TASK_ID_STACK_CUBE,
    TASK_ID_TRANSFER_CUBE,
]

TASK_CONFIGS = {
    TASK_ID_GRASP_CUBE: {
        'task_name': 'grasp_cube',
        'task_cls': GraspCubeTask,
        'ee_task_cls': GraspCubeEETask,
        'scripted_policy_cls': GraspCubePolicy,
        'episode_len': DEFAULT_EPISODE_LEN,
    },
    TASK_ID_STACK_CUBE: {
        'task_name': 'stack_cube',
        'task_cls': StackCubeTask,
        'ee_task_cls': StackCubeEETask,
        'scripted_policy_cls': StackCubePolicy,
        'episode_len': DEFAULT_EPISODE_LEN,
    },
    TASK_ID_TRANSFER_CUBE: {
        'task_name': 'transfer_cube',
        'task_cls': TransferCubeTask,
        'ee_task_cls': TransferCubeEETask,
        'scripted_policy_cls': TransferCubePolicy,
        'episode_len': DEFAULT_EPISODE_LEN,
    },
}