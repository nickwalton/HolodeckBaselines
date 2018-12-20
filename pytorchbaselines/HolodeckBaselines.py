from main import main
from main import show_model
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
import torch


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
    args.eval_interval = 20
    args.save_interval = 20
    args.cuda = False
    args.use_gae = True
    args.eval_render = True
    args.num_steps = 100
    args.log_interval = 10

    main(args)


def play_model(env_name, model_name):
    render_env = make_vec_envs(env_name, 1, 1,
                               None, None, False, device='cpu',
                               allow_early_resets=False)

    actor_critic, _ = torch.load(model_name)
    show_model(render_env, actor_critic, iterations=100)


if __name__ == '__main__':
    bipedal()

    #play_model('BipedalWalker-v2', 'trained_models/ppo/BipedalWalker-v2-AvgRwrd-114.pt')
