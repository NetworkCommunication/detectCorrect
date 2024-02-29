import time
from datetime import datetime

import Class_vehicle as Vehicle
import Class_rsu_0413 as RSU
import Class_DataCenter0413 as DC

if __name__ == "__main__":

    vehicle = Vehicle.Vehicles()
    rsu = RSU.RSU()
    dc = DC.DataCenter()
    # test_dataset = '.\\data_test_errordata.csv'
    test_dataset = 'bsm.csv'


    vehicle_time = vehicle.deal_messages(test_dataset)
    # i = 0
    s_time = time.time()
    dealCount = 0
    dealTotalTime = 0
    corrNumber = {}
    for t in vehicle_time:
        print("当前系统时间:",t)
        start_time = time.time()
        bsm_t = vehicle.send_messages(t)
        rsu.deal_record(bsm_t)
        for v_id in bsm_t.keys():
            print("v_id:", v_id)
            if v_id not in corrNumber.keys():
                corrNumber[v_id] = [1, 'T', 1, 0, 't']

            if corrNumber[v_id][1] == 'T':
                corr_project = rsu.process_record(v_id, t)
                corrNumber[v_id][3] = corr_project
                if all(corrNumber[v_id][3][i] == 0 for i in range(4)):
                    pass
                else:
                    corrNumber[v_id][4] = 'f'
                if corrNumber[v_id][4] == 'f':
                    corrNumber[v_id][0] = corrNumber[v_id][0] + 1

                if corrNumber[v_id][0] % 5 == 0:
                    corrNumber[v_id][1] = 'F'

                vehicle.receive_pro(v_id, corrNumber[v_id][3])
            else:
                corrNumber[v_id][2] = corrNumber[v_id][2] + 1
                if corrNumber[v_id][2] % 10 == 0:
                    temp = corrNumber[v_id][3]
                    corr_project = rsu.process_record(v_id, t)
                    if all(x == y for x, y in zip(temp, corr_project)):
                        pass
                    else:
                        corrNumber[v_id][1] = 'T'


        end_time = time.time()
        dealCount = dealCount + 1
        currentDealTime = end_time - start_time
        dealTotalTime = dealTotalTime + currentDealTime

    print("Average processing time of data once:", dealTotalTime/dealCount)