// target_model.forward(next_state)：這是 Double DQN 的關鍵，它用於估計下一狀態的 Q 值。通過引入一個目標網絡來分離動作選擇和 Q 值的計算，可以減少過估計的問題。
// update_target_model()：這個函數在每次訓練後更新目標網絡，這樣可以讓目標網絡的更新更加穩定，從而改善 Q-learning 的穩定性。
// epsilon：這是控制探索與利用平衡的參數，隨著訓練的進行，epsilon 會逐漸減少，模型會越來越傾向於利用學到的策略。

#include "DQNTrainer.h"

DQNTrainer::DQNTrainer(int input_size, int output_size, float gamma, float lr)
   : policy_model(input_size, output_size), optimizer(policy_model.parameters(), lr), gamma(gamma), target_model(input_size, output_size) {
   epsilon = 1.0;         // 初始化 epsilon，開始時有較高的探索率
   epsilon_decay = 0.995; // 每次訓練後減少探索率，這部分是為了避免過凝合，未來繼續調整，我還沒動
   epsilon_min = 0.01;    // 設定最小探索率，就只是定義一下，不然等等完全不探索卡在那。
   max_gred_norm = 10.0;  // 梯度下降的最大范數
   try 
   {
      policy_model.load_weights("dqn_model_weights.pt");
      target_model.load_weights("dqn_target_model_weights.pt");
      std::cout << "成功加載模型權重" << std::endl;
   } 
   catch (...) 
   {
      std::cout << "未找到保存的模型權重，從頭開始訓練。" << std::endl;
   }

   update_target_model();
}

void DQNTrainer::train(std::vector<Experience>& replay_memory) {
   if (replay_memory.empty()) return;

   policy_model.train();

   // 從經驗回放中採樣

   const int batch_size = replay_memory.size();
   std::vector<torch::Tensor> states, next_states, actions, rewards, dones;

   for (const auto& experience : replay_memory) {
       states.push_back(experience.state);
       next_states.push_back(experience.next_state);
       actions.push_back(torch::tensor(experience.action, torch::kLong));
       rewards.push_back(torch::tensor(experience.reward));
       dones.push_back(torch::tensor(experience.done ? 0.0f : 1.0f));
   }

   auto state_batch = torch::cat(states);
   auto next_state_batch = torch::cat(next_states);
   auto action_batch = torch::stack(actions);
   auto reward_batch = torch::stack(rewards);
   auto done_batch = torch::stack(dones);

   // 計算當前 Q 值
   auto q_values = policy_model.forward(state_batch);
   q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1);

   // 使用策略網路選擇下一步動作
   auto next_q_values_policy = policy_model.forward(next_state_batch);
   auto next_actions = next_q_values_policy.argmax(1);

   // 使用目標網路評估下一步 Q 值
   auto next_q_values_target = target_model.forward(next_state_batch);
   auto next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1);

   // 計算目標 Q 值
   auto target_q_values = reward_batch + gamma * next_q_values * done_batch;

   // 計算損失
   auto loss = torch::mse_loss(q_values, target_q_values.detach());

   optimizer.zero_grad();
   loss.backward();
   // 增加梯度下降
   torch::nn::utils::clip_grad_norm_(policy_model.parameters(), max_grad_norm);
   optimizer.step();

   update_epsilon();
   update_target_model(); // =更新目標網路

   policy_model.save_weights("dqn_policy_model_weights.pt");
   target_model.save_weights("dqn_target_model_weights.pt");
   std::cout << "模型權重以保存" << std::endl;
}

int DQNTrainer::select_action(torch::Tensor state) {
   policy_model.eval();
   if (torch::rand({1}).item<float>() < epsilon) {
      // 隨機選擇動作已增加探索性
      return torch::randint(0, 3, {1}).item<int>();
   } else {
      // 利用策略網路選擇最佳動作
      auto q_values = policy_model.forward(state);
      return q_values.argmax(1).item<int>();
   }
}

// 前向傳播，取得 Q 值
torch::Tensor DQNTrainer::forward(torch::Tensor state) {
   return policy_model.forward(state);
}

void DQNTrainer::update_epsilon() {
   if (epsilon > epsilon_min) {
      epsilon *= epsilon_decay; // 隨著訓練的進行減少探索率
      if (epsilon < epsilon_min) {
          epsilon = epsilon_min;
      }
      std::cout << "更新 epsilon 值：" << epsilon << std::endl;
   }
}

void DQNTrainer::update_target_model() {
   // 软更新参数
   float tau = 0.01;
   auto policy_params = policy_model.named_parameters();
   auto target_params = target_model.named_parameters();

   for (auto& item : policy_params) {
       auto& name = item.key();
       auto& param = item.value();
       auto& target_param = target_params[name];
       target_param.data().copy_(tau * param.data() + (1 - tau) * target_param.data());
   }
}
