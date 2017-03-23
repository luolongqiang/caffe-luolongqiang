import sys
import os
import os.path as osp
import argparse

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..')
sys.path.insert(0, caffe_path)
import caffe
caffe.set_mode_gpu()

def Solve(solver_path, gid = 0, model_path = None):
    print solver_path, gid
    if gid == None:
        gid = 0
    caffe.set_device(int(gid))
    solver = caffe.SGDSolver(solver_path)
    if model_path != None:
        solver.net.copy_from(model_path)
    solver.solve()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='solver_path')
    parser.add_argument('-i', dest='gid')
    parser.add_argument('-m', dest='model_path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    Solve(args.solver_path, args.gid, args.model_path)
