import sys
import matplotlib.pyplot as plt
import numpy as np
import time
arg_list = sys.argv
log_name = arg_list[1]
start = int(arg_list[2])
end = int(arg_list[3])
while True:
    with open(log_name,'r') as f:
        loss_list = []
        for line in f:
            i_begin = line.find("val_loss: ")
            if i_begin<0:
                continue
            sub = line[i_begin+10:i_begin+17]
            sub = float(sub)
            print(sub)
            loss_list.append(sub)
        length = len(loss_list)
        start = max(0, start)
        end = min(end, length)
        loss = loss_list[start:end]
        x=[i for i in range(start, end)]
        print(len(x))
        print("drawing...")
        plt.plot(x,loss)
        plt.savefig("historyplot.jpg", dpi=150)
        print("done")
    break

