CFLAGS= -I/usr/local/include/opencv -I.
LDFLAGS = -L/usr/local/lib -L.
OPENCVLIBS = -lopencv_core -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_nonfree

all: main.cpp 
	g++ -o EDLinesTest main.cpp -lEDLinesLib $(CFLAGS) $(LDFLAGS) $(OPENCVLIBS)


clean:
	rm -rf EDLinesTest core
