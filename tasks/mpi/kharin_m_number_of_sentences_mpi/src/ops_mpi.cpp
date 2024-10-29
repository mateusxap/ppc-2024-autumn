#include "mpi/kharin_m_number_of_sentences_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
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
  int base_part_size = 0;
  int remainder = 0;
  int text_length = 0;
  std::string full_text;  
  if (world.rank() == 0) {
    full_text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    text_length = full_text.size();
    base_part_size = text_length / world.size();
    remainder = text_length % world.size();
  }
  boost::mpi::broadcast(world, full_text, 0);
  boost::mpi::broadcast(world, base_part_size, 0);
  boost::mpi::broadcast(world, remainder, 0);
  int start = world.rank() * base_part_size + std::min(world.rank(), remainder);
  int end = start + base_part_size + (world.rank() < remainder ? 1 : 0);
  // Теперь используем разосланный текст
  local_text = full_text.substr(start, end - start);
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