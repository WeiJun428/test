#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/attr_value_util.h>
#include <tensorflow/core/platform/load_library.h>
#include <iostream>

#define LIB_DIR  ("/app/test-dir/tf_cuda_ops/build/")
#define FILENAME ((char* )"/tmp/tmpnrn54to4/tf_frozen.pb")

#define NUM_RUNS            1
#define VALID_BATCH_SIZE    1
#define MAX_NBBOX           256
#define NUM_INPUT_CHANNELS  1 // can be changed

using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::pair;

using tensorflow::Tensor;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::TensorShape;

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
    cerr << "Error reading " << file_name << ": " << Status.ToString() << ".\n";
    return false;
  }

  Session = tensorflow::NewSession(tensorflow::SessionOptions());
  if (Session == nullptr) {
    cerr << "Error creating Tensorflow session.\n";
    return false;
  }

  // Add the graph to the session
  Status = Session->Create(GraphDef);
  if (!Status.ok()) {
    cerr << "Error creating graph: " << Status.ToString() << ".\n";
    return false;
  }

  return true;
}

/**
 * Inspect every nodes in the graph
 */
void PrintGraph() {
  cout << "\nPrinting every nodes of the graph\n";
  
  const uint SIZE = GraphDef.node_size();
  for (int i = 0; i < SIZE; i++) {
    const auto node = GraphDef.node(i);
    cout << "Node " << i << ": " << node.name() << " (" << node.op() << ")\n";

    for (const auto& i : node.input()) {
      cout << "- input: " << i << "\n";
    }

    const auto attr = node.attr();
    for (const auto& x : attr) {
      cout << "- " << x.first << ": ";
      cout << SummarizeAttrValue(x.second) << "\n";
    }
    cout << "\n";
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
    tensorflow::Status Status = Session->Run(Input, { "output" }, {}, &Output);

    cout << "Done Inference " << i << std::endl;

    if (!Status.ok()) {
      cerr << "Error predicting " << Status.ToString() << "\n";
      return;
    }

    auto output_c = Output[0].tensor<float, 2>();

    /* 
     * Print tensor.
     *
     * auto output_c = Output[0].tensor<float, 2>();
     * for (int i = 0; i < 100; i++) {
     *  std::cout << output_c(0, i) << "\n";
     * } 
     */

    cout << Output[0].DebugString() << "\n";
  }
}

/**
 * Return random float [0, 1)
 */
float GetRand() {
  return static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
}

/**
 * Return model inputs
 */
vector<pair<string, Tensor>> GetDvDetInputs(int BatchSize, int BboxPadding, int NumInputChannels) {
  vector<int> NumOfList(BatchSize);
  const int LOW = 5;
  const int HIGH = 30;
  
  int NumPoints = 0;
  for (int i = 0; i < BatchSize; i++) {
    NumOfList[i] = rand() % (HIGH - LOW) + LOW;
    NumPoints += NumOfList[i];
    //cout << NumOfList[i] << "\n";
  }

  // Inputs
  Tensor Coors(DT_FLOAT, TensorShape({NumPoints, 3}));
  Tensor Features(DT_FLOAT, TensorShape({NumPoints, NumInputChannels}));
  Tensor NumList(DT_INT32, TensorShape({1, BatchSize}));
  Tensor IsTraining(false);

  Tensor Bboxes(DT_FLOAT, TensorShape({NumPoints, BboxPadding, 9}));

  float* CoorsData = Coors.flat<float>().data();
  for (int i = 0; i < NumPoints * 3; i++) {
    CoorsData[i] = GetRand();
  }

  float* FeaturesData = Features.flat<float>().data();
  for (int i = 0; i < NumPoints * NumInputChannels; i++) {
    FeaturesData[i] = GetRand();
  }

  cout << "1";

  int32_t* NumListData = NumList.flat<int32_t>().data();
  for (int i = 0; i < BatchSize; i++) {
    NumListData[i] = NumOfList[i];
  }

  cout << "2";

  float* BboxesData = Bboxes.flat<float>().data();
  for (int i = 0; i < NumPoints * BboxPadding * 9; i++) {
    BboxesData[i] = GetRand();
  }

  // Input and Output tensor
  vector<pair<string, Tensor>> Inputs = {
          {"stage1_input_coors_p", Coors},
          {"stage1_input_features_p", Features},
          {"stage1_input_num_list_p", NumList}, 
	  {"is_training", IsTraining}};
          //{"bboxes", Bboxes}};

  return Inputs;
}

/**
 * Inference the DvDet graph
 */
void PredictDvDet() {
  srand(0);
  // Input and Output tensor

  std::vector<tensorflow::Tensor> Output;

  for (int i = 0; i < NUM_RUNS; i++) {
    auto Input = GetDvDetInputs(VALID_BATCH_SIZE, MAX_NBBOX, NUM_INPUT_CHANNELS);

    // Replace "output" with tensor name
    tensorflow::Status Status = Session->Run(Input, { "ArgMax" }, {}, &Output);

    std::cout << "Done Inference " << i << std::endl;

    if (!Status.ok()) {
      std::cerr << "Error: " << Status.ToString() << "\n";
      return;
    }

    // auto output_c = Output[0].tensor<float, 2>();
    //std::cout << Output[0].DebugString() << "\n";
    Output.clear();
  }
}


/**
 * Import the custom ops library (.so) 
 * Return true if successful, false otherwise
 */
bool ImportLibrary() {
  string so_file[8] = {
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
    cout << "Loading library " << so_file[i] << "\n";
    string file_name = LIB_DIR + so_file[i];

    Status = tensorflow::internal::LoadLibrary(file_name.c_str(), &handler);
    if (!Status.ok()) {
      cerr << "Error: " << Status.ToString() << "\n";
      return false;
    }
  }

  return true;
}

int main() {
  if (ImportLibrary() && LoadGraph(FILENAME)) {
    PrintGraph();
    // Predict();
    PredictDvDet();
  }

  return 0;
}
