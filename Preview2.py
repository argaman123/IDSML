from DatasetHandler import *
from MachineLearning import *
from DataManagement import *
import Log as log

dTypes={'Dst Port':np.float32,'Protocol':np.float32,'Flow Duration':np.float32,'Tot Fwd Pkts':np.float32,'Tot Bwd Pkts':np.float32,'TotLen Fwd Pkts':np.float32,'TotLen Bwd Pkts':np.float32,'Fwd Pkt Len Max':np.float32,'Fwd Pkt Len Min':np.float32,'Fwd Pkt Len Mean':np.float32,'Fwd Pkt Len Std':np.float32,'Bwd Pkt Len Max':np.float32,'Bwd Pkt Len Min':np.float32,'Bwd Pkt Len Mean':np.float32,'Bwd Pkt Len Std':np.float32,'Flow Byts/s': np.str,'Flow Pkts/s':np.str,'Flow IAT Mean':np.float32,'Flow IAT Std':np.float32,'Flow IAT Max':np.float32,'Flow IAT Min':np.float32,'Fwd IAT Tot':np.float32,'Fwd IAT Mean':np.float32,'Fwd IAT Std':np.float32,'Fwd IAT Max':np.float32,'Fwd IAT Min':np.float32,'Bwd IAT Tot':np.float32,'Bwd IAT Mean':np.float32,'Bwd IAT Std':np.float32,'Bwd IAT Max':np.float32,'Bwd IAT Min':np.float32,'Fwd PSH Flags':np.float32,'Bwd PSH Flags':np.float32,'Fwd URG Flags':np.float32,'Bwd URG Flags':np.float32,'Fwd Header Len':np.float32,'Bwd Header Len':np.float32,'Fwd Pkts/s':np.float32,'Bwd Pkts/s':np.float32,'Pkt Len Min':np.float32,'Pkt Len Max':np.float32,'Pkt Len Mean':np.float32,'Pkt Len Std':np.float32,'Pkt Len Var':np.float32,'FIN Flag Cnt':np.float32,'SYN Flag Cnt':np.float32,'RST Flag Cnt':np.float32,'PSH Flag Cnt':np.float32,'ACK Flag Cnt':np.float32,'URG Flag Cnt':np.float32,'CWE Flag Count':np.float32,'ECE Flag Cnt':np.float32,'Down/Up Ratio':np.float32,'Pkt Size Avg':np.float32,'Fwd Seg Size Avg':np.float32,'Bwd Seg Size Avg':np.float32,'Fwd Byts/b Avg':np.float32,'Fwd Pkts/b Avg':np.float32,'Fwd Blk Rate Avg':np.float32,'Bwd Byts/b Avg':np.float32,'Bwd Pkts/b Avg':np.float32,'Bwd Blk Rate Avg':np.float32,'Subflow Fwd Pkts':np.float32,'Subflow Fwd Byts':np.float32,'Subflow Bwd Pkts':np.float32,'Subflow Bwd Byts':np.float32,'Init Fwd Win Byts':np.float32,'Init Bwd Win Byts':np.float32,'Fwd Act Data Pkts':np.float32,'Fwd Seg Size Min':np.float32,'Active Mean':np.float32,'Active Std':np.float32,'Active Max':np.float32,'Active Min':np.float32,'Idle Mean':np.float32,'Idle Std':np.float32,'Idle Max':np.float32,'Idle Min':np.float32,'Label':np.str}

dataHandler = DatasetHandler(filename="150218IDS.csv", numrows=250000, trainAmount=0.8, normalText="Benign",resultsColumn="Label",
                             dataTypes=dTypes, droppedColumns=['Timestamp', 'Flow Byts/s', 'Flow Pkts/s'])
mlwrapper = TempMLWrapper(dataHandler, 15)

log.show("main", "finished all preparation needed to test the machine, saving the progress")
save("ML", mlwrapper)
mlloaded = load("ML")

mlloaded.test()
mlloaded.preview()

