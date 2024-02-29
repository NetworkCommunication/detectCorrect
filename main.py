import time

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
    for t in vehicle_time:
        print("current time:",t)
        start_time = time.time()
        bsm_t = vehicle.send_messages(t)
        rsu.deal_record(bsm_t)
        for v_id in bsm_t.keys():
            corr_project = rsu.process_record(v_id, t)
            # vehicle.receive_pro(v_id, corr_project)
        end_time = time.time()

        end_time = time.time()
        dealCount = dealCount + 1
        currentDealTime = end_time - start_time
        dealTotalTime = dealTotalTime + currentDealTime

print("Average processing time of data once:", dealTotalTime / dealCount)