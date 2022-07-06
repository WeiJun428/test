#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/attr_value_util.h>
#include <tensorflow/core/platform/load_library.h>
#include <iostream>

#define FILENAME ((char* )"/tmp/tmpnrn54to4/tf_frozen.pb")

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

  for (int i = 0; i < 10; i++) {

    float* XData = X.flat<float>().data();
    for (int j = 0; j < 200; j++) {
      XData[i] = 1.0f * j + i;
    }

    // Replace "output" with tensor name
    tensorflow::Status Status = Session->Run(Input, { "ArgMax" }, {}, &Output);

    std::cout << "Done Inference " << i << std::endl;

    if (!Status.ok()) {
      std::cerr << "Error predicting " << Status.ToString() << "\n";
      return;
    }

    auto output_c = Output[0].tensor<float, 2>();
    /*
    for (int i = 0; i < 100; i++) {
      std::cout << output_c(0, i) << "\n";
    } */

    std::cout << Output[0].DebugString() << "\n";
  }
}

void PredictDvDet() {

  // Calculate number of points
  // Complicated configs to study tests/t.py

  // Inputs 
  tensorflow::Tensor Coors(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 200}));
  tensorflow::Tensor Features(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 200}));
  tensorflow::Tensor NumList(tensorflow::DT_INT32, tensorflow::TensorShape({1, 200}));
  tensorflow::Tensor Bboxes(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 200}));

  // Input and Output tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> Input = {
	  {"coors", Coors},
	  {"features", Features}, 
	  {"num_list", NumList},
  	  {"bboxes", Bboxes}};
  std::vector<tensorflow::Tensor> Output;

  for (int i = 0; i < 10; i++) {

    float* XData = X.flat<float>().data();
    for (int j = 0; j < 200; j++) {
      XData[i] = 1.0f * j + i;
    }

    // Replace "output" with tensor name
    tensorflow::Status Status = Session->Run(Input, { "output" }, {}, &Output);

    std::cout << "Done Inference " << i << std::endl;

    if (!Status.ok()) {
      std::cerr << "Error predicting " << Status.ToString() << "\n";
      return;
    }

    auto output_c = Output[0].tensor<float, 2>();
    /*
    for (int i = 0; i < 100; i++) {
      std::cout << output_c(0, i) << "\n";
    } */

    std::cout << Output[0].DebugString() << "\n";
  }
}


int main() {
  std::string so_file[8] = {
    "get_roi_bbox.so",
    "grid_sampling.so",
    "nms.so",
    "roi_logits_to_attrs.so",
    "voxel_sampling_idx.so",
    "radix_sort1d.so",
    "voxel_sampling_feature.so",
    "voxel_sampling_idx_binary.so"
  };
  tensorflow::Status Status;
  void* handler;

  for (int i = 0; i < 8; i++) {
    std::cout << "Loading library " << so_file[i] << "\n";
    std::string file_name = "/app/test-dir/tf_cuda_ops/build/" + so_file[i];
    Status = tensorflow::internal::LoadLibrary(file_name.c_str(), &handler);
    if (!Status.ok()) {
      std::cerr << "Error: " << Status.ToString() << "\n";
      return -1;
    }
  }

  if (LoadGraph(FILENAME)) {
    PrintGraph();
    Predict();
    // PredictDvDet();
  }

  return 0;
}
