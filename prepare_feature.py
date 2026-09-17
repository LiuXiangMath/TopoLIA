import argparse
from configs import Para
from src import prepare_laplacian



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataname", type=int, default=2016)
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    para = Para()
    prepare_laplacian(args.dataname,para)






