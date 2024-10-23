import subprocess
from argparse import ArgumentParser
from utils import *



if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--train_distribution",type=str,default='BA_20',help='Train distribution (if train and test are not the same)')
    parser.add_argument("--algorithm", type=str,default='EO', help="Algorithms")
    parser.add_argument("--test_distribution", type=str,default='BA_20', help="Distribution of training dataset")

    ## TS & EO
    parser.add_argument("--num_repeat", type=int,default=50, help="Number of attempts") ## TS and EO
    parser.add_argument("--num_steps", type=int, default=40, help="Number of steps")
    

    args = parser.parse_args()



    if args.algorithm=='EO' or args.algorithm=='TS':

        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'
    
    elif args.algorithm=='Greedy':

         command=f'python solvers/{args.algorithm}/evaluate.py --test_distribution {args.test_distribution} --num_repeat {args.num_repeat} '

    elif args.algorithm =='CAC' or args.algorithm =='AHC':
        command=f'python solvers/{args.algorithm}/evaluate.py --test_distribution {args.test_distribution}'

    elif args.algorithm =='RUN-CSP':
        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'

    elif args.algorithm == 'ANYCSP':
        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'

    elif args.algorithm == 'Gflow-CombOpt':
        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} --num_repeat {args.num_repeat}'

    elif args.algorithm == 'S2V-DQN':
        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} '

    elif args.algorithm == 'ECO-DQN' or args.algorithm == 'SoftTabu' or args.algorithm == 'LS-DQN':
        command=f'python solvers/{args.algorithm}/evaluate.py --train_distribution {args.train_distribution} --test_distribution {args.test_distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps} '



    else:
        raise ValueError('')
    
    subprocess.run(command,shell=True)




    
    





