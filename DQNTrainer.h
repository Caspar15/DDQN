// 先稍微說明我更改啥
// Experience：這是一個結構體，用於存儲經驗回放中的每一步，包括狀態、動作、獎勵和下一步的狀態。就是我們卡的 state 和 action 的部分。
// DQNTrainer：這是訓練 DQN 模型的核心類，包含了模型的更新、動作選擇等邏輯。
// target_model：這是一個新的模型，用於實現 Double DQN，這可以幫助減少 Q-learning 中的過估計問題。這部分很重要，還沒完全調整 Double DQN 的東西，我只是先把一個原理拿來試做，可以刪除。
// update_target_model()：這個函數負責將當前模型的權重複製到目標模型中，以實現 Double DQN。
#ifndef DQN_TRAINER_H
#define DQN_TRAINER_H

#include <torch/torch.h>
#include "DQNModel.h"
#include <vector>
#include <deque>

// 定義一個經驗回放數據結構，存儲每一步的狀態、動作、獎勵等，state 跟 action 就是這裡定義的。
struct Experience {
   torch::Tensor state;
   torch::Tensor next_state;
   int action;
   float reward;
   bool done; // 判斷是否為終止狀態
};

class DQNTrainer {
public:
   DQNTrainer(int input_size, int output_size, float gamma, float lr);
   void train(std::vector<Experience>& replay_memory);
   int select_action(torch::Tensor state);
   torch::Tensor forward(torch::Tensor state);
   void update_epsilon();  // 更新 epsilon，用於控制探索和利用的平衡
   // 我這邊說明一下我想到的方法，因為 DQN 是有回饋的，所以加上 greedy algorithm 的部分去跑，而 DQN 內的 貪心算法就是 epsilon 的定義。
   // 一開始選擇最優的 action，後續選擇較多的隨機動作，等結果跑出來再回測是哪個部分出現問題。

// 另一種方法
// public:
//    DQNTrainer(int input_size, int output_size, float gamma, float lr);
//    void train();
//    int select_action(torch::Tensor state);
//    torch::Tensor forward(torch::Tensor state);
//    void update_epsilon();
//    void store_experience(const Experience& exp);

// 這邊都是定義 epsilon 的部分
private:
   DQN policy_model; // 行為網路
   DQN target_model; // 用於 Double DQN 的目標網絡
   torch::optim::Adam optimizer; 
   float gamma;
   float epsilon;  // 探索率
   float epsilon_decay;  // 探索率衰減
   float epsilon_min;  // 最小探索率
   void update_target_model(); // 更新目標網絡的參數
   float max_gred_norm; // 梯度下降的最大范數

   // 另一種方法，經驗回放緩衝區
   // std::deque<Experience> replay_memory;
   // size_t replay_memory_capacity;
};

#endif // DQN_TRAINER_H