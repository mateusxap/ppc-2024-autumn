#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vladimirova_j_max_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> CreateVector(size_t size, size_t spread_of_val) {
  // Init value for input and output
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> CreateInputMatrix(size_t row_c, size_t col_c, size_t spread_of_val) {
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = CreateVector(col_c, spread_of_val);
  }
  return m;
}

TEST(vladimirova_j_max_of_vector_elements_mpi, test_pipeline_run) {
  int row = 7000;
  int col = 7000;
  int spread = 7000;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)spread + 10));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 random(dev());

    global_matrix = CreateInputMatrix(row, col, spread);
    int some_row = random() % row;
    int some_column = random() % col;
    global_matrix[some_row][some_column] = spread;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(spread, global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, test_task_run) {
  int row = 7000;
  int col = 7000;
  int spread = 7000;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)spread + 10));
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 random(dev());

    global_matrix = CreateInputMatrix(row, col, spread);
    int some_row = random() % row;
    int some_column = random() % col;
    global_matrix[some_row][some_column] = spread;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(spread, global_max[0]);
  }
}
