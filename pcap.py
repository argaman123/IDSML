import tkinter as tk
from scapy.all import *
from os import path
import subprocess
from MachineLearning import *
from DataManagement import *
#from scapy.arch.windows.compa import *
"""Adds stop_filter to sniff because its not in scapy-python3"""
import threading

dTypes={'Dst Port':np.float32,'Protocol':np.float32,'Flow Duration':np.float32,'Tot Fwd Pkts':np.float32,'Tot Bwd Pkts':np.float32,'TotLen Fwd Pkts':np.float32,'TotLen Bwd Pkts':np.float32,'Fwd Pkt Len Max':np.float32,'Fwd Pkt Len Min':np.float32,'Fwd Pkt Len Mean':np.float32,'Fwd Pkt Len Std':np.float32,'Bwd Pkt Len Max':np.float32,'Bwd Pkt Len Min':np.float32,'Bwd Pkt Len Mean':np.float32,'Bwd Pkt Len Std':np.float32,'Flow Byts/s': np.str,'Flow Pkts/s':np.str,'Flow IAT Mean':np.float32,'Flow IAT Std':np.float32,'Flow IAT Max':np.float32,'Flow IAT Min':np.float32,'Fwd IAT Tot':np.float32,'Fwd IAT Mean':np.float32,'Fwd IAT Std':np.float32,'Fwd IAT Max':np.float32,'Fwd IAT Min':np.float32,'Bwd IAT Tot':np.float32,'Bwd IAT Mean':np.float32,'Bwd IAT Std':np.float32,'Bwd IAT Max':np.float32,'Bwd IAT Min':np.float32,'Fwd PSH Flags':np.float32,'Bwd PSH Flags':np.float32,'Fwd URG Flags':np.float32,'Bwd URG Flags':np.float32,'Fwd Header Len':np.float32,'Bwd Header Len':np.float32,'Fwd Pkts/s':np.float32,'Bwd Pkts/s':np.float32,'Pkt Len Min':np.float32,'Pkt Len Max':np.float32,'Pkt Len Mean':np.float32,'Pkt Len Std':np.float32,'Pkt Len Var':np.float32,'FIN Flag Cnt':np.float32,'SYN Flag Cnt':np.float32,'RST Flag Cnt':np.float32,'PSH Flag Cnt':np.float32,'ACK Flag Cnt':np.float32,'URG Flag Cnt':np.float32,'CWE Flag Count':np.float32,'ECE Flag Cnt':np.float32,'Down/Up Ratio':np.float32,'Pkt Size Avg':np.float32,'Fwd Seg Size Avg':np.float32,'Bwd Seg Size Avg':np.float32,'Fwd Byts/b Avg':np.float32,'Fwd Pkts/b Avg':np.float32,'Fwd Blk Rate Avg':np.float32,'Bwd Byts/b Avg':np.float32,'Bwd Pkts/b Avg':np.float32,'Bwd Blk Rate Avg':np.float32,'Subflow Fwd Pkts':np.float32,'Subflow Fwd Byts':np.float32,'Subflow Bwd Pkts':np.float32,'Subflow Bwd Byts':np.float32,'Init Fwd Win Byts':np.float32,'Init Bwd Win Byts':np.float32,'Fwd Act Data Pkts':np.float32,'Fwd Seg Size Min':np.float32,'Active Mean':np.float32,'Active Std':np.float32,'Active Max':np.float32,'Active Min':np.float32,'Idle Mean':np.float32,'Idle Std':np.float32,'Idle Max':np.float32,'Idle Min':np.float32,'Label':np.str}


# --- functions ---
"""
def sniffing():
	print('DEBUG: before sniff')
	#sniff(filter="host 192.168.0.48", prn=action, stop_filter=stop_sniffing)
	sniff(prn=action, stop_filter=stop_sniffing)
	print('DEBUG: after sniff')

def action(packet):
	try:
		print("%s went to %s" % (packet[IP].src, packet[IP].dst))
	except Exception as e:
		print(e)
"""

def stop_sniffing(x):
	global switch
	return switch

# ---

def start_button():
	global switch
	global thread
	global fileout
	if (thread is None) or (not thread.is_alive()):
		switch = False
		thread = threading.Thread(target=create_pcap, daemon=True)
		fileout = open("results.csv", "a+")
		thread.start()
	else:
		print('DEBUG: already running')

def stop_button():
	global switch

	print('DEBUG: stoping')

	switch = True
	fileout.close()


def create_pcap():
	#open pcap file and write packets
	print('DEBUG: operating #1')
	os.chdir(r"./CICFlowMeter-master" )
	while True:
		print('DEBUG: operating #2')
		packets = sniff(timeout=20, stop_filter=stop_sniffing) #still need to select interface
		wrpcap("data/packets.pcap", packets)
		p = subprocess.call('gradlew exeCMD --args="data/packets.pcap data/out/"', shell=True) 
		dataset = Dataset(filename="data/out/packets.pcap_Flow.csv", resultsColumn="Label", dataTypes=dTypes, droppedColumns=["Flow ID","Src IP","Src Port","Dst IP"])
		try: # in case the loop was closed while running
			for row in dataset:
				fileout.write(f"{','.join(str(e) for e in row)}, {'Benign' if mlp.isNormal(row) else 'Attack'}\n")
				print(f"converted one row")
		except IOError:
			return
		os.remove("data/out/packets.pcap_Flow.csv")


def exec_csv():
	if stop == True:
		return
	os.chdir( r"/CICFlowMeter-master" )
	os.system('cmd /k "gradlew exeCMD --args="data/packets.pcap data/out/""')
# --- main ---

mlp :PartialMLWrapper = load("MLP")


thread = None 
switch = False
stop=False

root = tk.Tk()

tk.Button(root, text="Start sniffing", command=start_button).pack()
tk.Button(root, text="Stop sniffing", command=stop_button).pack()

root.mainloop()

#deleting the file is problematic

#delte file before anouther itteration
#check threading