// ops_mpi.cpp
#include "mpi/kharin_m_seidel_method/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>

namespace mpi = boost::mpi;

bool kharin_m_seidel_method::GaussSeidelSequential::pre_processing() {
  internal_order_test();

  // Чтение eps из taskData
  eps = *(reinterpret_cast<double*>(taskData->inputs[1]));

  a.resize(n * n);
  b.resize(n);
  x.resize(n, 1.0);  // Инициализация x значением 1.0
  p.resize(n);

  // Чтение матрицы A из taskData->inputs[2]
  auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
  std::copy(a_data, a_data + n * n, a.begin());

  // Чтение вектора b из taskData->inputs[3]
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
  std::copy(b_data, b_data + n, b.begin());

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
    // Чтение матрицы A из taskData->inputs[2]
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);

    // Проверка условия сходимости и единственности решения
    for (int i = 0; i < n; ++i) {
      double diag = std::abs(a_data[i * n + i]);
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += std::abs(a_data[i * n + j]);
        }
      }
      if (diag <= sum || a_data[i * n + i] == 0.0) {
        std::cerr
            << "Матрица A не является строго диагонально доминантной или имеет нулевой элемент на диагонали в строке "
            << i << ".\n";
        is_valid = false;
        break;
      }
    }
  }
  return is_valid;
}

bool kharin_m_seidel_method::GaussSeidelSequential::run() {
  internal_order_test();

  bool converged = false;
  int m = 0;

  while (!converged && m < max_iterations) {
    // Копирование x в p
    p = x;

    // Обновление x
    for (int i = 0; i < n; i++) {
      double var = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != i) {
          var += a[i * n + j] * x[j];
        }
      }
      x[i] = (b[i] - var) / a[i * n + i];
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
  std::copy(x.begin(), x.end(), x_output);
  return true;
}

bool kharin_m_seidel_method::GaussSeidelParallel::pre_processing() {
  internal_order_test();

  // Процесс 0 считывает данные
  if (world.rank() == 0) {
    eps = *(reinterpret_cast<double*>(taskData->inputs[1]));
    // Инициализация векторов
    a.resize(n * n);
    b.resize(n);
    x.resize(n, 1.0);  // Инициализация x значением 1.0
    p.resize(n);

    // Чтение матрицы A
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
    std::copy(a_data, a_data + n * n, a.begin());

    // Чтение вектора b
    auto* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
    std::copy(b_data, b_data + n, b.begin());

    // Инициализация x
    for (int i = 0; i < n; i++) {
      x[i] = 1.0;
    }
  }

  // Распространение n и eps
  mpi::broadcast(world, n, 0);
  mpi::broadcast(world, eps, 0);
  mpi::broadcast(world, max_iterations, 0);

  if (world.rank() != 0) {
    // Инициализация векторов
    a.resize(n * n);
    b.resize(n);
    x.resize(n, 1.0);
    p.resize(n);
  }
  // Распространение матрицы A и вектора b
  mpi::broadcast(world, a.data(), n * n, 0);
  mpi::broadcast(world, b.data(), n, 0);
  return true;
}

bool kharin_m_seidel_method::GaussSeidelParallel::validation() {
  internal_order_test();
  bool is_valid = true;

  if (world.rank() == 0) {
    n = *(reinterpret_cast<int*>(taskData->inputs[0]));
    // Проверка размеров входных данных
    if (taskData->inputs_count[0] != static_cast<size_t>(1) || taskData->inputs_count[1] != static_cast<size_t>(1) ||
        taskData->inputs_count[2] != static_cast<size_t>(n * n) ||
        taskData->inputs_count[3] != static_cast<size_t>(n) || taskData->outputs_count[0] != static_cast<size_t>(n)) {
      is_valid = false;
    }

    if (is_valid) {
      // Чтение матрицы A из taskData->inputs[2]
      auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);

      // Проверка условия сходимости и единственности решения
      for (int i = 0; i < n; ++i) {
        double diag = std::abs(a_data[i * n + i]);
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
          if (j != i) {
            sum += std::abs(a_data[i * n + j]);
          }
        }
        if (diag <= sum || a_data[i * n + i] == 0.0) {
          std::cerr
              << "Матрица A не является строго диагонально доминантной или имеет нулевой элемент на диагонали в строке "
              << i << ".\n";
          is_valid = false;
          break;
        }
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
  int m = 0;

  while (!converged && m < max_iterations) {
    // Копирование x в p
    p = x;

    // Обновление x для своих строк
    for (int i = start; i < end; i++) {
      double var = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != i) {
          var += a[i * n + j] * x[j];
        }
      }
      x[i] = (b[i] - var) / a[i * n + i];
    }

    // Объединение обновленных значений x
    for (int i = 0; i < world.size(); i++) {
      int s = i * n / world.size();
      int e = (i + 1) * n / world.size();
      mpi::broadcast(world, &x[s], e - s, i);
    }

    // Вычисление локальной нормы
    double local_norm = 0.0;
    for (int i = start; i < end; i++) {
      local_norm += (x[i] - p[i]) * (x[i] - p[i]);
    }

    // Вычисление глобальной нормы
    double global_norm = 0.0;
    mpi::all_reduce(world, local_norm, global_norm, std::plus<>());

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
    auto* x_output = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(x.begin(), x.end(), x_output);
  }

  return true;
}