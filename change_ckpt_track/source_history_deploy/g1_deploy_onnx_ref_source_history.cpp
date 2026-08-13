/**
 * Isolated SONIC deploy wrapper for qpos-track source-history-prefilled rollouts.
 *
 * NVIDIA deployment sources remain untouched.  This translation unit derives
 * a StateLogger adapter, inserts the nine source snapshots before the first
 * CONTROL tick, and substitutes the source current-frame last_action in that
 * first live logger entry.  All subsequent behavior is the official deploy.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp"

namespace source_history_prefill {

constexpr std::size_t kHistoryEntries = 9;
constexpr std::size_t kBodyJoints = 29;

std::string prefill_file;

template <std::size_t N>
std::array<double, N> ReadArray(const nlohmann::json& object,
                                const char* field) {
  const auto& value = object.at(field);
  if (!value.is_array() || value.size() != N) {
    throw std::runtime_error(
        std::string("source-history field '") + field + "' must have " +
        std::to_string(N) + " values");
  }
  std::array<double, N> result{};
  for (std::size_t index = 0; index < N; ++index) {
    result[index] = value.at(index).get<double>();
    if (!std::isfinite(result[index])) {
      throw std::runtime_error(
          std::string("source-history field '") + field +
          "' contains a non-finite value");
    }
  }
  return result;
}

template <std::size_t N>
double MaxAbsError(const std::span<double>& actual,
                   const std::array<double, N>& expected) {
  if (actual.size() != N) {
    return std::numeric_limits<double>::infinity();
  }
  double error = 0.0;
  for (std::size_t index = 0; index < N; ++index) {
    error = std::max(error, std::abs(actual[index] - expected[index]));
  }
  return error;
}

template <std::size_t N>
double MaxAbsError(const std::array<double, N>& actual,
                   const std::array<double, N>& expected) {
  double error = 0.0;
  for (std::size_t index = 0; index < N; ++index) {
    error = std::max(error, std::abs(actual[index] - expected[index]));
  }
  return error;
}

struct Snapshot {
  int policy_seq = -1;
  int source_row_index = -1;
  std::array<double, 4> base_quat{};
  std::array<double, 3> base_ang_vel{};
  std::array<double, kBodyJoints> body_q{};
  std::array<double, kBodyJoints> body_dq{};
  std::array<double, kBodyJoints> last_action{};
};

Snapshot ReadSnapshot(const nlohmann::json& object) {
  Snapshot result;
  result.policy_seq = object.at("policy_seq").get<int>();
  result.source_row_index = object.at("source_row_index").get<int>();
  result.base_quat = ReadArray<4>(object, "base_quat");
  result.base_ang_vel = ReadArray<3>(object, "base_ang_vel");
  result.body_q = ReadArray<kBodyJoints>(object, "body_q");
  result.body_dq = ReadArray<kBodyJoints>(object, "body_dq");
  result.last_action = ReadArray<kBodyJoints>(object, "last_action");
  return result;
}

}  // namespace source_history_prefill

class SourceHistoryStateLogger final : public StateLogger {
 public:
  SourceHistoryStateLogger(
      std::string csv_dir, size_t ring_capacity, int num_joints = -1,
      int num_actions = -1, double dt_seconds = 0.0, bool enable_csv = true,
      std::map<std::string, std::variant<std::string, int, double, bool>>
          robot_config = {})
      : StateLogger(std::move(csv_dir), ring_capacity, num_joints, num_actions,
                    dt_seconds, enable_csv, std::move(robot_config)) {
    if (!source_history_prefill::prefill_file.empty()) {
      LoadPrefill(source_history_prefill::prefill_file);
    }
  }

  uint64_t LogFullState(
      const std::array<double, 4>& base_quat,
      const std::array<double, 3>& base_ang_vel,
      const std::array<double, 3>& base_accel,
      const std::array<double, 4>& body_torso_quat,
      const std::array<double, 3>& body_torso_ang_vel,
      const std::array<double, 3>& body_torso_accel,
      const std::span<double>& body_q, const std::span<double>& body_dq,
      const std::span<double>& last_action,
      const std::span<double>& motor_temperature,
      const std::span<double>& motor_error,
      const std::span<double>& motor_torque,
      const std::span<double>& left_hand_q,
      const std::span<double>& left_hand_dq,
      const std::span<double>& right_hand_q,
      const std::span<double>& right_hand_dq,
      const std::span<double>& last_left_hand_action,
      const std::span<double>& last_right_hand_action,
      double ros_timestamp = 0.0) {
    if (!first_live_pending_) {
      return StateLogger::LogFullState(
          base_quat, base_ang_vel, base_accel, body_torso_quat,
          body_torso_ang_vel, body_torso_accel, body_q, body_dq, last_action,
          motor_temperature, motor_error, motor_torque, left_hand_q,
          left_hand_dq, right_hand_q, right_hand_dq, last_left_hand_action,
          last_right_hand_action, ros_timestamp);
    }

    const double quat_error = source_history_prefill::MaxAbsError(
        base_quat, expected_current_.base_quat);
    const double angular_velocity_error = source_history_prefill::MaxAbsError(
        base_ang_vel, expected_current_.base_ang_vel);
    const double q_error = source_history_prefill::MaxAbsError(
        body_q, expected_current_.body_q);
    const double dq_error = source_history_prefill::MaxAbsError(
        body_dq, expected_current_.body_dq);
    std::cout << "[SourceHistoryPrefill] first-live alignment: policy_seq="
              << expected_current_.policy_seq
              << ", source_row=" << expected_current_.source_row_index
              << ", max_abs(quat/w/q/dq)=" << quat_error << "/"
              << angular_velocity_error << "/" << q_error << "/" << dq_error
              << std::endl;

    // q/orientation is initialized to this exact phase.  DDS conversion and
    // interpolation justify a slightly looser velocity tolerance.
    if (quat_error > 1e-4 || q_error > 1e-4 ||
        angular_velocity_error > 2e-3 || dq_error > 2e-3) {
      throw std::runtime_error(
          "source-history current state does not match the first live DDS "
          "state; refusing a misaligned qpos-track experiment");
    }

    first_live_pending_ = false;
    std::span<double> source_last_action(expected_current_.last_action);
    return StateLogger::LogFullState(
        base_quat, base_ang_vel, base_accel, body_torso_quat,
        body_torso_ang_vel, body_torso_accel, body_q, body_dq,
        source_last_action, motor_temperature, motor_error, motor_torque,
        left_hand_q, left_hand_dq, right_hand_q, right_hand_dq,
        last_left_hand_action, last_right_hand_action, ros_timestamp);
  }

 private:
  void LoadPrefill(const std::string& path) {
    std::ifstream stream(path);
    if (!stream.good()) {
      throw std::runtime_error("cannot open source-history prefill file: " + path);
    }
    nlohmann::json payload;
    stream >> payload;
    if (payload.at("format").get<std::string>() !=
            "g1_decoder_source_history_prefill" ||
        payload.at("version").get<int>() != 1) {
      throw std::runtime_error("unsupported source-history prefill format/version");
    }
    if (payload.at("history_order").get<std::string>() !=
        "oldest_to_newest") {
      throw std::runtime_error("source-history entries must be oldest_to_newest");
    }
    const auto& entries = payload.at("entries");
    if (!entries.is_array() ||
        entries.size() != source_history_prefill::kHistoryEntries ||
        payload.at("history_entry_count").get<std::size_t>() !=
            source_history_prefill::kHistoryEntries) {
      throw std::runtime_error("source-history prefill must contain exactly 9 entries");
    }

    std::array<double, 3> zero3{};
    std::array<double, 4> zero4{};
    std::array<double, 7> zero7{};
    std::array<double, 29> zero29{};
    std::array<double, 58> zero58{};
    int previous_policy_seq = -1;
    for (const auto& object : entries) {
      auto snapshot = source_history_prefill::ReadSnapshot(object);
      if (previous_policy_seq >= 0 &&
          snapshot.policy_seq != previous_policy_seq + 1) {
        throw std::runtime_error("source-history policy_seq is not contiguous");
      }
      previous_policy_seq = snapshot.policy_seq;

      // Call the base implementation deliberately: these are old snapshots,
      // not live CONTROL ticks.  They enter the normal GetLatest ring.
      StateLogger::LogFullState(
          snapshot.base_quat, snapshot.base_ang_vel, zero3, zero4, zero3,
          zero3, std::span<double>(snapshot.body_q),
          std::span<double>(snapshot.body_dq),
          std::span<double>(snapshot.last_action),
          std::span<double>(zero58), std::span<double>(zero29),
          std::span<double>(zero29), std::span<double>(zero7),
          std::span<double>(zero7), std::span<double>(zero7),
          std::span<double>(zero7), std::span<double>(zero7),
          std::span<double>(zero7), 0.0);
    }

    expected_current_ = source_history_prefill::ReadSnapshot(payload.at("current"));
    if (expected_current_.policy_seq != previous_policy_seq + 1) {
      throw std::runtime_error(
          "source-history current policy_seq does not follow the prefill entries");
    }
    first_live_pending_ = true;
    std::cout << "[SourceHistoryPrefill] loaded " << entries.size()
              << " source entries from " << path << "; current policy_seq="
              << expected_current_.policy_seq << ", source_row="
              << expected_current_.source_row_index << std::endl;
  }

  bool first_live_pending_ = false;
  source_history_prefill::Snapshot expected_current_{};
};

// Compile the official controller against the adapter.  The custom option is
// removed first, so the official parser receives its normal argv exactly.
#define StateLogger SourceHistoryStateLogger
#define main g1_deploy_onnx_ref_official_main
#include "../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
#undef main
#undef StateLogger

int main(int argc, char const* argv[]) {
  std::vector<char const*> forwarded_argv;
  forwarded_argv.reserve(static_cast<std::size_t>(argc));
  forwarded_argv.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) != "--source-history-prefill-file") {
      forwarded_argv.push_back(argv[index]);
      continue;
    }
    if (!source_history_prefill::prefill_file.empty()) {
      std::cerr << "Error: --source-history-prefill-file was supplied more than once"
                << std::endl;
      return 2;
    }
    if (index + 1 >= argc) {
      std::cerr << "Error: --source-history-prefill-file requires a path"
                << std::endl;
      return 2;
    }
    source_history_prefill::prefill_file = argv[index + 1];
    ++index;
  }
  return g1_deploy_onnx_ref_official_main(
      static_cast<int>(forwarded_argv.size()), forwarded_argv.data());
}
