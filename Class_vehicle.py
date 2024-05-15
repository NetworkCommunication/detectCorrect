class Vehicles:
    corr_project = {}
    bsms = {}
    """bsms={t:{id:[t,x,y,v,a,...]}}"""

    def deal_messages(self, filepath):  # '.\\data_test2_err.csv'
        """
         Put the vehicle information in the file in the vehicle class
         :param filepath: file address to save vehicle information
         :return: vehicle information time
         """
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.split(",")
                re_time = line[1]
                if re_time not in self.bsms.keys():
                    self.bsms[re_time] = {}
                vid = line[0]
                x = float(line[2])
                y = float(line[3])
                v = float(line[7])
                a = float(line[8])
                self.bsms[re_time][vid] = [re_time, x, y, v, a]
        f.close()
        return self.bsms.keys()

    def send_messages(self, now_time):
        """
         Read the current time and return the vehicle information corresponding to the time
         :param now_time: time
         :return: Vehicle information corresponding to the time. If the correction plan has been obtained before, the corrected information will be returned. Otherwise, the pre-corrected information will be returned.
         """
        t = str(now_time)
        for vid in self.bsms[t]:
            if vid in self.corr_project.keys():
                self.bsms[t][vid] = [self.bsms[t][vid][0]] + [self.bsms[t][vid][i + 1] + self.corr_project[vid][i]
                                    for i in range(4)]
                print("vid:",vid, ";corr_project:",self.corr_project[vid])
            else:
                self.corr_project[vid] = [0.0, 0.0, 0.0, 0.0]
        return self.bsms[str(t)]

    def receive_pro(self, v_id: str, corr_project: list):
        if all(corr_project[i] == 0.0 for i in range(len(corr_project))):
            pass
        else:
            self.corr_project[v_id] = corr_project
