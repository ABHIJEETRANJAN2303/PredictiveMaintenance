import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from collections import deque

PORT = "/dev/cu.usbserial-110"
BAUD = 115200

WINDOW_SIZE = 300

ser = serial.Serial(PORT, BAUD, timeout=1)

app = QtWidgets.QApplication([])

window = pg.GraphicsLayoutWidget(title="Live Accelerometer")

plot = window.addPlot(title="Acceleration")

plot.showGrid(x=True, y=True)

plot.addLegend()

x_data = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
y_data = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
z_data = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)

curve_x = plot.plot(pen='r', name="X")
curve_y = plot.plot(pen='g', name="Y")
curve_z = plot.plot(pen='b', name="Z")

window.show()


def update():

    while ser.in_waiting:

        line = ser.readline().decode(errors="ignore").strip()

        parts = line.split(",")

        if len(parts) != 3:
            continue

        ax = float(parts[0])
        ay = float(parts[1])
        az = float(parts[2])

        x_data.append(ax)
        y_data.append(ay)
        z_data.append(az)

    curve_x.setData(list(x_data))
    curve_y.setData(list(y_data))
    curve_z.setData(list(z_data))


timer = QtCore.QTimer()

timer.timeout.connect(update)

timer.start(20)

app.exec()

ser.close()