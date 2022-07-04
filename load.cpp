#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/attr_value_util.h>
#include <iostream>

#define FILENAME ((char* )"/tmp/tmpsm0lo99w/tf_frozen.pb")

// Global graph definition and session
tensorflow::GraphDef GraphDef;
tensorflow::Session* Session = nullptr;

/**
 * Load the graph of given .pb filename and populate the gloval GraphDef and Session.
 */
bool LoadGraph(char* file_name) {
  // Read the graph
  tensorflow::Status Status = ReadBinaryProto(tensorflow::Env::Default(), file_name, &GraphDef);
  if (!Status.ok()) {
    std::cerr << "Error reading graph definition from " << file_name << ": " << Status.ToString() << ".\n";
    return false;
  }

  Session = tensorflow::NewSession(tensorflow::SessionOptions());
  if (Session == nullptr) {
    std::cerr << "Error creating Tensorflow session.\n";
    return false;
  }

  // Add the graph to the session
  Status = Session->Create(GraphDef);
  if (!Status.ok()) {
    std::cerr << "Error creating graph: " << Status.ToString() << ".\n";
    return false;
  }

  return true;
}

/**
 * Inspect every nodes in the graph
 */
void PrintGraph() {
  std::cout << "\nPrinting every nodes of the graph\n";
  
  const uint SIZE = GraphDef.node_size();
  for (int i = 0; i < SIZE; i++) {
    const auto node = GraphDef.node(i);
    std::cout << "Node " << i << ": " << node.name() << " (" << node.op() << ")\n";

    for (const auto& i : node.input()) {
      std::cout << "- input: " << i << "\n";
    }

    const auto attr = node.attr();
    for (const auto& x : attr) {
      std::cout << "- " << x.first << ": ";
      std::cout << SummarizeAttrValue(x.second) << "\n";
    }
    std::cout << "\n";
  }
}

void Predict() {
  tensorflow::Tensor X(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 200}));
  std::vector<std::pair<std::string, tensorflow::Tensor>> Input = {{"input", X}};
  std::vector<tensorflow::Tensor> Output;

  float* XData = X.flat<float>().data();
  for (int i = 0; i < 200; i++) {
    XData[i] = 1.0f * i;
  }

  // Replace YOUR_OUTPUT_TENSOR with your output tensor name
  tensorflow::Status Status = Session->Run(Input, { "output" }, {}, &Output);

  std::cout << "Done Inference\n";

  if (!Status.ok()) {
    std::cerr << "Error predicting " << Status.ToString() << "\n";
    return;
  }

  auto output_c = Output[0].tensor<float, 2>();

  for (int i = 0; i < 100; i++) {
    std::cout << output_c(0, i) << "\n";
  }

  std::cout << Output[0].DebugString() << "\n";
  // std::cout << Output.size() << "\n";
}


int main() {
  if (LoadGraph(FILENAME)) {
    PrintGraph();
    Predict();
  }

  return 0;
}
