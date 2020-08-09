import os
import sys

if __name__=="__main__":

    folds = ['fold_1' , 'fold_2' , 'fold_3' , 'fold_4' , 'fold_5']
    maneuver_type = sys.argv[1]
    interval_time = sys.argv[2]
    if maneuver_type == 'all':
        index = ['356988', '356988', '356988', '356988', '356988']
    elif maneuver_type == 'all_new_features':
        index = ['846483', '846483', '846483', '846483', '846483']
    elif maneuver_type == 'lane':
        index = ['723759', '723759', '723759', '723759', '723759']
    elif maneuver_type == 'turns':
        index = ['209221', '209221', '209221', '209221', '209221']
    for i,f in zip(index,folds):
        i += interval_time
        print(i,f)
        os.system('THEANO_FLAGs=mode=FAST_RUN,device=gpu,floatX=float32 python ./maneuver-rnn.py {0} {1} {2}'.format(i,f,maneuver_type))
