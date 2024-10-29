#include "mpi/kharin_m_number_of_sentences_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>

namespace kharin_m_number_of_sentences_mpi {

int CountSentences(const std::string& text) {
  int count = 0;
  for (size_t i = 0; i < text.size(); i++) {
    char c = text[i];
    if (c == '.' || c == '?' || c == '!') {
      count++;
    }
  }
  return count;
}

bool CountSentencesSequential::pre_processing() {
  internal_order_test();
  text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  sentence_count = 0;
  return true;
}

bool CountSentencesSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool CountSentencesSequential::run() {
  internal_order_test();
  sentence_count = CountSentences(text);
  return true;
}

bool CountSentencesSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  return true;
}

bool CountSentencesParallel::pre_processing() {
  internal_order_test();
  // Переменные для деления текста
  int text_length = 0;
  int base_part_size = 0;
  int remainder = 0;
  std::vector<std::string> text_parts(world.size());  // Вектор для подстрок текста
  if (world.rank() == 0) {
    // Только процесс с рангом 0 инициализирует текст и делит его на части
    text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    text_length = text.size();
    base_part_size = text_length / world.size();
    remainder = text_length % world.size();
    // Разделяем текст на подстроки для каждого процесса
    int start = 0;
    for (int i = 0; i < world.size(); i++) {
      int end = start + base_part_size + (i < remainder ? 1 : 0);
      text_parts[i] = text.substr(start, end - start);
      start = end;
    }
  }
  // Используем scatter для распределения частей текста
  boost::mpi::scatter(world, text_parts, local_text, 0);
  // Инициализируем счетчик предложений
  sentence_count = 0;
  return true;
}

bool CountSentencesParallel::validation() {
  internal_order_test();
  return world.rank() == 0 ? taskData->outputs_count[0] == 1 : true;
}

bool CountSentencesParallel::run() {
  internal_order_test();
  // Подсчет предложений в локальной части текста
  int local_count = CountSentences(local_text);
  // Суммирование результатов
  boost::mpi::reduce(world, local_count, sentence_count, std::plus<>(), 0);
  return true;
}

bool CountSentencesParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  }
  return true;
}

}  // namespace kharin_m_number_of_sentences_mpi