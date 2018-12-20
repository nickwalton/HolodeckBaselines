from main import main
from a2c_ppo_acktr.arguments import get_args


def cartpole():
    args = get_args()
    args.env_name = 'CartPole-v0'
    args.algo = 'ppo'
    args.no_cuda = True
    args.eval_interval = 10
    args.eval_render = True

    main(args)


def bipedal():
    args = get_args()
    args.env_name = 'BipedalWalker-v2'
    args.algo = 'ppo'
    args.eval_interval = 1000
    args.eval_render = True
    args.num_processes = 8

    main(args)


if __name__ == '__main__':
    bipedal()
