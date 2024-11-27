// ops_mpi.cpp
#include "mpi/kharin_m_seidel_method/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <memory>

namespace mpi = boost::mpi;

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
  double* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i][j] = a_data[i * n + j];
    }
  }

  // Чтение вектора b из taskData->inputs[3]
  double* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
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
  n = *(reinterpret_cast<int*>(taskData->inputs[0]));
  // Проверка количества входных и выходных данных
  if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
    return false;
  }
  // Проверка размеров входных данных
  else if (taskData->inputs_count[0] != static_cast<size_t>(1) ||      // n
           taskData->inputs_count[1] != static_cast<size_t>(1) ||      // eps
           taskData->inputs_count[2] != static_cast<size_t>(n * n) ||  // Матрица A
           taskData->inputs_count[3] != static_cast<size_t>(n) ||      // Вектор b
           taskData->outputs_count[0] != static_cast<size_t>(n)) {     // Вектор x
    return false;

  // Проверка условия сходимости
  double* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
  for (int i = 0; i < n; ++i) {
    double diag = std::abs(a_data[i * n + i]);
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (j != i) {
        sum += std::abs(a_data[i * n + j]);
      }
    }
    if (diag <= sum) {
      return false;
    }
  }

  return true;
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
  double* x_output = reinterpret_cast<double*>(taskData->outputs[0]);
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

bool kharin_m_seidel_method::GaussSeidelParallel::pre_processing() {
  internal_order_test();

  // Процесс 0 считывает данные
  if (world.rank() == 0) {
    eps = *(reinterpret_cast<double*>(taskData->inputs[1]));

    // Выделение памяти
    a = new double*[n];
    for (int i = 0; i < n; i++) {
      a[i] = new double[n];
    }
    b = new double[n];
    x = new double[n];
    p = new double[n];

    // Чтение матрицы A
    double* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        a[i][j] = a_data[i * n + j];
      }
    }

    // Чтение вектора b
    double* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
    for (int i = 0; i < n; i++) {
      b[i] = b_data[i];
    }

    // Инициализация x
    for (int i = 0; i < n; i++) {
      x[i] = 1.0;
    }
  }

  // Распространение n и eps
  mpi::broadcast(world, n, 0);
  mpi::broadcast(world, eps, 0);

  if (world.rank() != 0) {
    // Выделение памяти
    a = new double*[n];
    for (int i = 0; i < n; i++) {
      a[i] = new double[n];
    }
    b = new double[n];
    x = new double[n];
    p = new double[n];
  }

  // Распространение матрицы A и вектора b
  for (int i = 0; i < n; i++) {
    mpi::broadcast(world, a[i], n, 0);
  }
  mpi::broadcast(world, b, n, 0);

  // Инициализация x на остальных процессах
  if (world.rank() != 0) {
    for (int i = 0; i < n; i++) {
      x[i] = 1.0;
    }
  }

  return true;
}

bool kharin_m_seidel_method::GaussSeidelParallel::validation() {
  internal_order_test();
  bool is_valid = true;

  if (world.rank() == 0) {
    n = *(reinterpret_cast<int*>(taskData->inputs[0]));
    // Проверка количества входных и выходных данных
    if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
      is_valid = false;
    }

    // Проверка размеров входных данных
    else if (taskData->inputs_count[0] != static_cast<size_t>(1) ||      // n
             taskData->inputs_count[1] != static_cast<size_t>(1) ||      // eps
             taskData->inputs_count[2] != static_cast<size_t>(n * n) ||  // Матрица A
             taskData->inputs_count[3] != static_cast<size_t>(n) ||      // Вектор b
             taskData->outputs_count[0] != static_cast<size_t>(n)) {     // Вектор x
      is_valid = false;
    }

    // Проверка условия сходимости
    double* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
    for (int i = 0; i < n; ++i) {
      double diag = std::abs(a_data[i * n + i]);
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += std::abs(a_data[i * n + j]);
        }
      }
      if (diag <= sum) {
        std::cerr << "Матрица A не является строго диагонально доминантной в строке " << i << ".\n";
        is_valid = false;
      }
    }
  }

  // Распространение результата проверки
  mpi::broadcast(world, n, 0);
  mpi::broadcast(world, is_valid, 0);

  return is_valid;
}

bool kharin_m_seidel_method::GaussSeidelParallel::run() {
  internal_order_test();

  int start = world.rank() * n / world.size();
  int end = (world.rank() + 1) * n / world.size();

  bool converged = false;
  int max_iterations = 10000;
  int m = 0;

  while (!converged && m < max_iterations) {
    // Копирование x в p
    for (int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    // Обновление x для своих строк
    for (int i = start; i < end; i++) {
      double var = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != i) {
          var += a[i][j] * x[j];
        }
      }
      x[i] = (b[i] - var) / a[i][i];
    }

    // Объединение обновленных значений x
    for (int i = 0; i < world.size(); i++) {
      int s = i * n / world.size();
      int e = (i + 1) * n / world.size();
      mpi::broadcast(world, x + s, e - s, i);
    }

    // Вычисление локальной нормы
    double local_norm = 0.0;
    for (int i = start; i < end; i++) {
      local_norm += (x[i] - p[i]) * (x[i] - p[i]);
    }

    // Вычисление глобальной нормы
    double global_norm = 0.0;
    mpi::all_reduce(world, local_norm, global_norm, std::plus<double>());

    converged = (sqrt(global_norm) < eps);

    // Распространение флага сходимости
    mpi::broadcast(world, converged, 0);

    m++;
  }

  return true;
}

bool kharin_m_seidel_method::GaussSeidelParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Запись результатов в taskData->outputs[0]
    double* x_output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (int i = 0; i < n; i++) {
      x_output[i] = x[i];
    }
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