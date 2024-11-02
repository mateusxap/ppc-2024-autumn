#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/vladimirova_j_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> CreateVector(size_t size, size_t spread_of_val) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> CreateInputMatrix(size_t row_c, size_t column_c, size_t spread_of_val) {
  //  Init value for input and output
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = CreateVector(column_c, spread_of_val);
  }
  return m;
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_ValMatrix_0) {
  const size_t size = 0;
  const int spread = 10;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), false);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_CanCreate_10) {
  const size_t col = 10;
  const size_t row = 10;
  const int spread = 10;
  EXPECT_NO_THROW(CreateInputMatrix(row, col, spread));
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_1) {
  const size_t size = 1;
  const int spread = 10;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  in[0][0] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_NotSquareMatrix_1_2) {
  const size_t col = 1;
  const size_t row = 2;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(row, col, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % row;
  in[some_row][0] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_NotSquareMatrix_3_1) {
  const size_t col = 3;
  const size_t row = 1;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(row, col, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_col = random() % col;
  in[0][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_10) {
  const size_t size = 10;
  const int spread = 10;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % size;
  int some_col = random() % size;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_20) {
  const size_t size = 20;
  const int spread = 50;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % size;
  int some_col = random() % size;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_50) {
  const size_t size = 50;
  const int spread = 50;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % size;
  int some_col = random() % size;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_100) {
  const size_t size = 100;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % size;
  int some_col = random() % size;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_SquareMatrix_100_WithSeveralMax) {
  const size_t size = 100;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(size, size, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % size;
  int some_col = random() % size;
  in[some_row][some_col] = spread;
  some_row = random() % size;
  some_col = random() % size;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_NotSquareMatrix_100_50_WithSeveralMax) {
  const size_t col = 100;
  const size_t row = 50;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -110);
  std::vector<std::vector<int>> in = CreateInputMatrix(row, col, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % row;
  int some_col = random() % col;
  in[some_row][some_col] = spread;
  some_row = random() % row;
  some_col = random() % col;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}

TEST(vladimirova_j_max_of_vector_elements_seq, Test_NotSquareMatrix_100_50) {
  const size_t col = 100;
  const size_t row = 50;
  const int spread = 100;  // spread is excepted answer

  // Create data
  std::vector<int> out(1, -((int)spread + 10));
  std::vector<std::vector<int>> in = CreateInputMatrix(row, col, spread);

  std::random_device dev;
  std::mt19937 random(dev());
  int some_row = random() % row;
  int some_col = random() % col;
  in[some_row][some_col] = spread;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  vladimirova_j_max_of_vector_elements_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(spread, out[0]);
}
