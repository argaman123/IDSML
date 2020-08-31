import tkinter as tk
from scapy.all import *
from os import path
import subprocess
#from scapy.arch.windows.compa import *
"""Adds stop_filter to sniff because its not in scapy-python3"""
import threading

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

    if (thread is None) or (not thread.is_alive()):
        switch = False
        thread = threading.Thread(target=create_pcap, daemon=True)
        thread.start()
    else:
        print('DEBUG: already running')

def stop_button():
    global switch

    print('DEBUG: stoping')

    switch = True


def create_pcap():
	#open pcap file and write packets
	print('DEBUG: operating #1')
	os.chdir( r"C:/Users/Hana/Desktop/workspace/degree/CICFlowMeter-master" )
	while True:
		print('DEBUG: operating #2')
		packets = sniff(timeout=10, stop_filter=stop_sniffing) #still need to select interface
		wrpcap("C:\\Users\\Hana\\Desktop\\workspace\\degree\\data\\packets.pcap", packets)
		subprocess.call('gradlew exeCMD --args="C:\\Users\\Hana\\Desktop\\workspace\\degree\\data\\packets.pcap C:\\Users\\Hana\\Desktop\\workspace\\degree\\data"', shell=True)

		#ARGAMAN your code goes here

def exec_csv():
	if stop == True:
		return
	os.chdir( r"C:/Users/Hana/Desktop/workspace/degree/CICFlowMeter-master" )
	os.system('cmd /k "gradlew exeCMD --args="C:\\Users\\Hana\\Desktop\\workspace\\degree\\data\\packets.pcap C:\\Users\\Hana\\Desktop\\workspace\\degree\\data""')
# --- main ---

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