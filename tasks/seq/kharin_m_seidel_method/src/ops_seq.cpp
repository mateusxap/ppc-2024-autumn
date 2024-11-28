// ops_mpi.cpp
#include "seq/kharin_m_seidel_method/include/ops_seq.hpp"

#include <cmath>

bool kharin_m_seidel_method::GaussSeidelSequential::pre_processing() {
  internal_order_test();

  // Чтение eps из taskData
  eps = *(reinterpret_cast<double*>(taskData->inputs[1]));

  // Выделение памяти для матрицы и векторов
  a = new double*[n];
  for (int i = 0; i < n; i++) {
    a[i] = new double[n];
  }
  b = new double[n];
  x = new double[n];
  p = new double[n];

  // Чтение матрицы A из taskData->inputs[2]
  auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i][j] = a_data[i * n + j];
    }
  }

  // Чтение вектора b из taskData->inputs[3]
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
  for (int i = 0; i < n; i++) {
    b[i] = b_data[i];
  }

  // Инициализация вектора x
  for (int i = 0; i < n; i++) {
    x[i] = 1.0;
  }

  return true;
}

bool kharin_m_seidel_method::GaussSeidelSequential::validation() {
  internal_order_test();
  bool is_valid = true;
  n = *(reinterpret_cast<int*>(taskData->inputs[0]));
  // Проверка размеров входных данных
  if (taskData->inputs_count[0] != static_cast<size_t>(1) || taskData->inputs_count[1] != static_cast<size_t>(1) ||
      taskData->inputs_count[2] != static_cast<size_t>(n * n) || taskData->inputs_count[3] != static_cast<size_t>(n) ||
      taskData->outputs_count[0] != static_cast<size_t>(n)) {
    is_valid = false;
  }

  if (is_valid) {
    // Проверка условия сходимости
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
    for (int i = 0; i < n; ++i) {
      double diag = std::abs(a_data[i * n + i]);
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += std::abs(a_data[i * n + j]);
        }
      }
      if (diag <= sum) {
        is_valid = false;
      }
    }
  }

  return is_valid;
}

bool kharin_m_seidel_method::GaussSeidelSequential::run() {
  internal_order_test();

  bool converged = false;
  int max_iterations = 10000;  // Максимальное количество итераций
  int m = 0;

  while (!converged && m < max_iterations) {
    // Копирование x в p
    for (int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    // Обновление x
    for (int i = 0; i < n; i++) {
      double var = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != i) {
          var += a[i][j] * x[j];
        }
      }
      x[i] = (b[i] - var) / a[i][i];
    }

    // Проверка сходимости
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
      norm += (x[i] - p[i]) * (x[i] - p[i]);
    }
    converged = (sqrt(norm) < eps);
    m++;
  }

  return true;
}

bool kharin_m_seidel_method::GaussSeidelSequential::post_processing() {
  internal_order_test();

  // Запись результатов в taskData->outputs[0]
  auto* x_output = reinterpret_cast<double*>(taskData->outputs[0]);
  for (int i = 0; i < n; i++) {
    x_output[i] = x[i];
  }

  // Освобождение памяти
  for (int i = 0; i < n; i++) {
    delete[] a[i];
  }
  delete[] a;
  delete[] b;
  delete[] x;
  delete[] p;

  return true;
}