#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace unitree_policy_bridge
{

// Mirrors unitree_rl_lab's isaaclab::ObservationTermCfg
// (deploy/include/isaaclab/manager/manager_term_cfg.h): clip-then-scale each element, then
// buffer the last `history_length` frames. Using the same semantics here means the same
// deploy.yaml produces the same observation vector as the C++ hardware deploy path.
struct ObservationTerm
{
  std::function<std::vector<float>()> compute;
  std::vector<float> scale;   // per-element; empty = no scaling
  std::vector<float> clip;    // [min, max], applied to every element; empty = no clipping
  int history_length = 1;

  void add(std::vector<float> obs)
  {
    for (size_t j = 0; j < obs.size(); ++j) {
      if (!clip.empty()) {
        obs[j] = std::clamp(obs[j], clip[0], clip[1]);
      }
      if (!scale.empty()) {
        obs[j] *= scale[j];
      }
    }
    buffer.push_back(std::move(obs));
    while (static_cast<int>(buffer.size()) > history_length) {
      buffer.pop_front();
    }
  }

  void reset(const std::vector<float> & obs)
  {
    buffer.clear();
    for (int i = 0; i < history_length; ++i) {
      add(obs);
    }
  }

  // Concatenates the buffered frames oldest-to-newest (the non-gym-history layout - the
  // default unless a robot's deploy.yaml sets use_gym_history: true, which isn't supported
  // here yet since none of the exported manifests seen so far set it).
  std::vector<float> get() const
  {
    std::vector<float> concatenated;
    for (const auto & frame : buffer) {
      concatenated.insert(concatenated.end(), frame.begin(), frame.end());
    }
    return concatenated;
  }

  std::deque<std::vector<float>> buffer;
};

}  // namespace unitree_policy_bridge
