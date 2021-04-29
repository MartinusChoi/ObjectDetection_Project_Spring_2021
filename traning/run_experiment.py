import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl

np.random.seed(42)
torch.manual_seed(42)

def _setup_parser():
    # ArgumentParser 객체 생성
    # => 명령행을 파이썬 데이터형으로 파싱하는데 필요한 모든 정보를 담고 있음.
    parser = argparse.ArgumentParser(add_help=False)

    # 인자 추가하기 => .add_argument() 호출
    # 이 호출은 일반적으로 ArgumentParser 객체에게 명령행의 문자열을 객체로 변환하는 방법을 알려줌.
    # 이 정보는 저장되며, parse_args() 가 호출될 때 사용됨.
    parser.add_argument("--name", type=str, default="youngmin")
    parser.add_argument("--age", type=int, default=20)
    parser.add_argument("--phone_number", type=str, default=None)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    """
    Run an experiment

    Sample command4
    '''
    python training/run_experiment.py --name choi --age 23 --phone_number 010-6231-3285
    '''
    """

    parser = _setup_parser()

    # 인자 파싱하기 => .parse_args() 호출
    # 이 메서드는 명령행을 검사하고 각 인자를 적절한 형으로 변환한 다음 액션을 호출함.
    # args.(인자 이름) 으로 해당 인자 값에 접근할 수 있음.
    args = parser.parse_args()
    
    print(f"name is {args.name}")
    print(f"age is {args.age}")
    print(f"phone_number is {args.phone_number}")

    if args.age != 23 :
        print("you are 23! idiot!")
    else:
        print('correct age')

    return

# 해당 python file의 인터프리터에서 실행하면 
# => __name__에 '__main__'이 담겨서 실행됨

# 다른 file의 인터프리터에서 해당 python 모듈을 임포트해서 사용하게 되면
#  => __name__에 해당 python file의 이름이 담겨서 실행됨.
if __name__ == '__main__':
    main()