import subprocess
from argparse import ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser()

    
    parser.add_argument("--algorithm", type=str,default='TS', help="Algorithms")
    parser.add_argument("--distribution", type=str,default='BA_20', help="Distribution of training dataset")

    ## TS & EO
    parser.add_argument("--num_repeat", type=int,default=50, help="Number of attempts") ## TS and EO
    parser.add_argument("--num_steps", type=int, default=40, help="Number of steps")
    
    parser.add_argument("--low", type=float, default=20, help="lower bound of parameter")
    parser.add_argument("--high", type=float, default=30, help="higher bound of parameter")
    parser.add_argument('--step',type=float, default=10, help="Step size of parameter")


    args = parser.parse_args()



    if args.algorithm=='EO' or args.algorithm=='TS':

        command=f'python solvers/{args.algorithm}/train.py --distribution {args.distribution} --num_repeat {args.num_repeat} --num_steps {args.num_steps} --low {args.low} --high {args.high} --step {args.step}'

    elif args.algorithm=='RUN-CSP':

        command=f'python solvers/{args.algorithm}/train.py --distribution {args.distribution} '

    elif args.algorithm=='ANYCSP':
        command=f'python solvers/{args.algorithm}/train.py --distribution {args.distribution} '




    else:
        raise ValueError('')
    
    subprocess.run(command,shell=True,check=True)




    
    





