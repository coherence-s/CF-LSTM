import numpy as np
import cPickle as cp
import sys

'''
Run this script after runbatchprediction.py in order to choose the best threshold and checkpoint for each fold.

Input: 
Path to the pickle file generated by runnbatchprediction.py
'''
if __name__ == "__main__":
    count = sys.argv[2] 
    results_file = sys.argv[1]
    all_res = np.zeros((int(count), 3)) 
    for index in range(0, int(count)):
        file_name = results_file + str(index) + '.pik'
        results = cp.load(open(file_name))

        threshold = results['threshold']
        checkpoints = results['checkpoints']

        p_mat = results['precision']
        re_mat = results['recall']
        time_mat = results['time']

        summary = []

        count_th = 0
        for th in threshold:
            count_checkpoint = 0
            for checkpoint in checkpoints:
                p = p_mat[count_th,count_checkpoint,-1]
                r = re_mat[count_th,count_checkpoint,-1]
                t = time_mat[count_th,count_checkpoint,-1]

                f1 = 2.0*p*r/(p+r)

                summary.append(np.array([p,r,f1,t,th,checkpoint]))

                count_checkpoint += 1

            count_th += 1

        summary = np.array(summary)

        arg_list = np.argsort(summary[:,2])

        reverse_list = list(arg_list)

        reverse_list =  np.array(reverse_list[::-1])

        summary = summary[reverse_list,:]

        print("[Precision, Recall, F1, Threshold, Checkpoint]")
        print(summary[0, :])
        all_res[index, :] = summary[0, 0:3]
    print ("\n\n")
    print ("FINAL RESULTS")
    print (np.mean(all_res, 0))
    print (np.max(all_res, 0)-np.mean(all_res, 0))
    print (np.mean(all_res, 0)-np.min(all_res, 0))