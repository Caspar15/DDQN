#include "DQNTrainer.h"

// 建構子：初始化模型、最佳化器等
DQNTrainer::DQNTrainer(int input_size, int output_size, float gamma, float lr, int target_update_freq)
    : model(input_size, output_size), optimizer(model.parameters(), torch::optim::AdamOptions(lr).weight_decay(1e-5)), gamma(gamma), target_model(input_size, output_size), target_update_frequency(target_update_freq)
{
   epsilon = 1.0;         // 初始化探索率 epsilon，開始時完全探索
   epsilon_decay = 0.995; // 每次訓練後減少 epsilon，逐步減少探索
   epsilon_min = 0.01;    // 設定最小探索率，防止探索率過低

   // 將模型移到 CUDA
   if (torch::cuda::is_available())
   {
      model.to(torch::kCUDA);        // 將主網路移到 GPU
      target_model.to(torch::kCUDA); // 將目標網路移到 GPU
      std::cout << "Using CUDA" << std::endl;
      std::cout << "CUDA Device Count: " << torch::cuda::device_count() << std::endl;
   }
   else
   {
      std::cout << "Using CPU" << std::endl;
   }

   // 初始化目標網絡
   update_target_model();

   // 引入學習率調度器，每訓練 100 步驟減少學習率 10%
   scheduler = std::make_shared<torch::optim::StepLR>(optimizer, /*step_size=*/100, /*gamma=*/0.1);
}

// Double DQN：訓練函數
void DQNTrainer::train(std::vector<Experience> &replay_memory, int train_step)
{
   model.train(); // 將模型置於訓練模式
   for (auto &experience : replay_memory)
   {
      // 確保所有張量在同一台裝置上（GPU）
      auto device = model.parameters().begin()->device();
      auto state = experience.state.to(device);                                                // 目前狀態
      auto next_state = experience.next_state.to(device);                                      // 下一狀態
      auto action = torch::tensor({experience.action}, torch::dtype(torch::kLong)).to(device); // 動作
      auto reward = torch::tensor({experience.reward}).to(device);                             // 獎勵

      // 目前狀態的 Q 值
      auto q_values = model.forward(state);                              // 計算目前狀態的 Q 值
      auto q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1); // 選取執行的動作對應的 Q 值

      // Ddouble DQN：使用主網路選擇動作
      auto next_q_values_online = model.forward(next_state);                      // 透過主網路計算下一狀態的 Q 值
      auto next_action_online = std::get<1>(torch::max(next_q_values_online, 1)); // 選擇 Q 值最大的動作

      // Double DQN: 使用目標網路估計 Q 值
      auto next_q_values_target = target_model.forward(next_state);
      auto max_next_q_value = next_q_values_target.gather(1, next_action_online.unsqueeze(1)).squeeze(1);

      // 計算目標值
      auto gamma_tensor = torch::tensor({gamma}).to(device);
      auto target = reward + gamma_tensor * max_next_q_value; // 目標 Q 值 = 目前獎勵 + 折扣後的最大 Q 值

      // 計算損失（使用目標 Q 值與目前 Q 值之間的均方誤差）
      auto loss = torch::mse_loss(q_value, target.detach()); // detach 是為了防止目標 Q 值參與反向傳播

      // 反向傳播和參數更新
      optimizer.zero_grad(); // 清除之前的梯度
      loss.backward();       // 計算梯度
      optimizer.step();      // 更新模型參數
   }

   // 更新 epsilon（探索率），逐步減少探索
   update_epsilon();

   // 控制目標網路更新頻率
   if (train_step % target_update_frequency == 0)
   {
      update_target_model();
   }

   // 更新學習率
   scheduler->step();
}

// 選擇動作：epsilon-greedy 策略
int DQNTrainer::select_action(torch::Tensor state)
{
   model.eval(); // 評估模式
   auto device = model.parameters().begin()->device();
   state = state.to(device); // 確保狀態在正確的裝置上 (GPU)

   if (torch::rand(1).item<float>() < epsilon) // 以 epsilon 的機率進行探索
   {
      // 探索：隨機選擇動作
      return torch::randint(0, 5, {1}, torch::TensorOptions().device(device)).item<int>();
   }
   else
   {
      // 利用：選擇 DQN 模型預測的最佳動作
      auto q_values = model.forward(state);
      return std::get<1>(torch::max(q_values, 1)).item<int>();
   }
}

// 直接前向傳播，用於評估
torch::Tensor DQNTrainer::forward(torch::Tensor state)
{
   auto device = model.parameters().begin()->device();
   return model.forward(state.to(device));
}

// 更新 epsilon 值，逐步减少探索率
void DQNTrainer::update_epsilon()
{
   if (epsilon > epsilon_min)
   {
      epsilon *= epsilon_decay;
   }
}

// 更新目標網路的參數
void DQNTrainer::update_target_model()
{
   // 使用無梯度模式複製參數，防止梯度傳播到目標網絡
   torch::NoGradGuard no_grad; // 禁用梯度計算
   auto model_params = model.named_parameters();
   auto target_params = target_model.named_parameters();
   for (const auto &item : model_params)
   {
      auto name = item.key();
      auto *target_param = target_params.find(name);
      if (target_param != nullptr)
      {
         target_param->copy_(item.value());
      }
   }
}

// 取得模型所在的設備
torch::Device DQNTrainer::get_device()
{
   return model.parameters().begin()->device();
}
