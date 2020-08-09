# -*- coding: UTF-8 -*-
import shutil,os

if __name__=="__main__":
    actions = ['lane', 'turns', 'all', 'all_new_features']   #设置要训练的任务
    #actions = ['lane', 'turns']
    for maneuver_type in actions:
        for i in range(0,10):    #重复进行10次训练
            for interval_time in range(13,27):  #延时时间
                shell0 = 'python delayTime.py '+ maneuver_type + ' ' +  str(interval_time)
                os.system(shell0)
                shell1 = 'python trainModels.py ' + maneuver_type + ' ' + str(interval_time)
                os.system(shell1)
                shell2 = 'python runbatchprediction.py '+ maneuver_type +'_zhouxq' + ' ' + str(interval_time)
                os.system(shell2)
                shell3 = 'cp ./checkpoints/'+ maneuver_type +'_zhouxq/complete_results_final_model.pik ./dong_models'
                os.system(shell3) 
                shutil.move("./dong_models/complete_results_final_model.pik", "./dong_models/complete_results_final_model_"+maneuver_type+str(interval_time) +str(i)+".pik")
