import re
import numpy as np
import matplotlib.pyplot as plt

fileLocation = "./"
fileName = "result_retrain.log"
# targetFileName = "log.csv"
pattern1 = '(?<=accuracy\s=\s).*'
pattern2 = '(?<= sparsity:\s).*'

class LogParser:
    def extractPattern(self, sFileLocation, sTargetFileName, sPattern1, sPattern2):
        with open(sFileLocation + fileName, mode="rt", encoding='utf-8') as f:
            string = f.read()
        
        label = np.expand_dims(np.arange(1,1.3,0.015), axis=1)
        label = label.astype('float32')
#         print(label)

        pattern1 = re.compile(sPattern1)
        pattern2 = re.compile(sPattern2)

        accu = pattern1.findall(string)
        acculist_1 = accu[1::3]
        acculist_2 = accu[2::3]
#         print(acculist_1)
#         print(acculist_2)

        acculist = np.column_stack((acculist_1,acculist_2))
        acculist = acculist.astype('float32')
        # print(acculist)

        spar = pattern2.findall(string)
        sparlist = spar[1::3]
        # print(sparlist)

        a = np.column_stack((acculist,sparlist))
        # print(a.dtype)
        result = np.column_stack((label, a))
        result = result.astype('float32')

        ## save as file
        # for i in result:
        #     with open(sFileLocation + sTargetFileName, mode="a", encoding='utf-8') as f:
        #         item = '    '.join(i)
        #         f.write(item+'\n')
        # print("success")

        return result

parser = LogParser()
result = parser.extractPattern(fileLocation, targetFileName, pattern1, pattern2)
