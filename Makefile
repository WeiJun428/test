CXX = g++ -o
HEADER_FLAG = -I/usr/local/tensorflow/include -L/usr/local/tensorflow/lib
LB_FLAG = -ltensorflow_cc -ltensorflow_framework
CUSTOM_OP_FLAG = #-lgrid_sampling
EXE = test
load:
	$(CXX) $(EXE) load.cpp $(HEADER_FLAG) $(LB_FLAG) $(CUSTOM_OP_FLAG)
test:
	$(CXX) $(EXE) test.cpp $(HEADER_FLAG) $(LB_FLAG)
clean:
	rm $(EXE)
