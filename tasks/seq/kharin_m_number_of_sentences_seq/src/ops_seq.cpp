#include "seq/kharin_m_number_of_sentences_seq/include/ops_seq.hpp"

#include <algorithm>
#include <string>

namespace kharin_m_number_of_sentences_seq {

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

}  // namespace kharin_m_number_of_sentences_seq
