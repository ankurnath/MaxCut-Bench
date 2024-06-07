import subprocess
from argparse import ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser()

    
    parser.add_argument("--algorithm", type=str,default='EO', help="Algorithms")
    parser.add_argument("--distribution", type=str,default='BA_20', help="Distribution of training dataset")

    ## TS & EO
    parser.add_argument("--num_repeat", type=int,default=50, help="Number of attempts") ## TS and EO
    parser.add_argument("--num_steps", type=int, default=40, help="Number of steps")
    

    args = parser.parse_args()



    if args.algorithm=='EO' or args.algorithm=='TS':

        command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'
    
    elif args.algorithm=='Greedy':

         command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution} --num_repeat {args.num_repeat} '

    elif args.algorithm =='CAC' or args.algorithm =='AHC':
        command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution}'

    elif args.algorithm =='RUN-CSP':
        command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'

    elif args.algorithm == 'ANYCSP':
        command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps}'

    elif args.algorithm == 'Gflownet-CombOpt':
        command=f'python solvers/{args.algorithm}/evaluate.py --distribution {args.distribution} --num_repeat {args.num_repeat}'



    else:
        raise ValueError('')
    
    subprocess.run(command,shell=True,check=True)




    
    





