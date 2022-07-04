CXX = g++ -o
HEADERFLAG = -I/usr/local/tensorflow/include -L/usr/local/tensorflow/lib
LBFLAG = -ltensorflow_cc -ltensorflow_framework
EXE = test
load:
	$(CXX) $(EXE) load.cpp $(HEADERFLAG) $(LBFLAG)
test:
	$(CXX) $(EXE) test.cpp $(HEADERFLAG) $(LBFLAG)
clean:
	rm $(EXE)
