// Golovkin Maksim

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_integration_rectangular_method/include/ops_seq.hpp"
using namespace golovkin_integration_rectangular_method;
using ppc::core::TaskData;

TEST(golovkin_integration_rectangular_method_seq, Calculate_ZeroToOne) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.01;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 1.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_ZeroToTwo) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0;
  double b = 2.0;
  double epsilon = 0.01;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 8.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_NegativeToPositive) {
  auto taskData = std::make_shared<TaskData>();
  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.01;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 2.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_LargeInterval_HighPrecision) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.00001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 333.3333333333333, 0.01);
}
TEST(golovkin_integration_rectangular_method_seq, Validation_AllConditionsMet) {
  auto taskData = std::make_shared<TaskData>();
  taskData->inputs_count = {2, 3};
  taskData->outputs_count = {2};
  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
}

TEST(golovkin_integration_rectangular_method_seq, Validation_InvalidInputsCountSize) {
  auto taskData = std::make_shared<TaskData>();
  taskData->inputs_count = {2};
  taskData->outputs_count = {2};
  IntegralCalculator calculator(taskData);

  ASSERT_FALSE(calculator.validation());
}

TEST(golovkin_integration_rectangular_method_seq, Validation_OutputInputCountMismatch) {
  auto taskData = std::make_shared<TaskData>();
  taskData->inputs_count = {2, 3};
  taskData->outputs_count = {1};
  IntegralCalculator calculator(taskData);

  ASSERT_FALSE(calculator.validation());
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_DegenerateInterval) {
  auto taskData = std::make_shared<TaskData>();
  double a = 1.0;
  double b = 1.0;
  double epsilon = 0.01;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 0.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_NegativeInterval) {
  auto taskData = std::make_shared<TaskData>();
  double a = 2.0;
  double b = 0.0;
  double epsilon = 0.01;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, -8.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_HighPrecisionSmallEpsilon) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0;
  double b = 1.0;
  double epsilon = 1e-6;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 1.0 / 3.0, 1e-4);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_LargeInterval) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0;
  double b = 100.0;
  double epsilon = 0.1;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_EQ(calculator.validation(), false);
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 333333.333, 100.0);
}

TEST(golovkin_integration_rectangular_method_seq, Validation_EdgeCaseInputCounts) {
  auto taskData = std::make_shared<TaskData>();
  taskData->inputs_count = {0, 1};
  taskData->outputs_count = {2};
  IntegralCalculator calculator(taskData);

  ASSERT_FALSE(calculator.validation());
}