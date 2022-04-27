import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')
ani_fig = plt.figure()
ax1 = ani_fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('HW5_plot_bus.csv','r').read()
    lines = graph_data.split('\n')
    xs = []
    y1s = []
    y2s = []
    # y3s = []
    for line in lines[1:]:
        if len(line) > 1:
            # batch_size, training_acc, testing_acc, training_time = line.split(',')
            data = line.split(',')
            xs.append(float(data[0]))
            y1s.append(float(data[1]))
            y2s.append(float(data[3]))
            # y3s.append(float(data[4]))
    ax1.clear()
    ax1.plot(xs, y1s, "-b", label="Loss")
    ax1.plot(xs, y2s, "-r", label="Cost")
    # ax1.plot(xs, [x/max(y3s) for x in y3s], "-y", label="Relative training time") #scale over max training time
    ax1.legend(loc="upper left")
    # plt.ylim(0, 1.5)



ani = animation.FuncAnimation(ani_fig, animate, interval=10)
plt.xlabel("steps")
# plt.ylabel("Loss")
plt.show()
