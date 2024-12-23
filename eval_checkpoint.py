import argparse
from train import eval_bc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Evaluation parameters
    parser.add_argument('--ckpt_path', action='store', type=str, help='path to checkpoint to evaluate', required=True)
    parser.add_argument('--eval_instr_path', action='store', type=str, help='path to the csv file containing instruction embeddings for evaluation')
    parser.add_argument('--instruction', action='store', type=str, help='path to the csv file containing instruction embeddings for evaluation')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='number of rollouts to perform during evaluation')
    parser.add_argument('--videos_dir', action='store', type=str, help='number of rollouts to perform during evaluation')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--temporal_agg', action='store_true')

    args = parser.parse_args()

    print("Starting...")
    eval_bc(
        ckpt_path=args.ckpt_path,
        instr_path=args.eval_instr_path,
        instr_text=args.instruction,
        num_rollouts=args.num_rollouts,
        videos_dir=args.videos_dir,
        onscreen_render=args.onscreen_render,
        temporal_agg=args.temporal_agg,
    )
