// torch::nn::Linear：PyTorch 的線性層，fc1 到 fc4 是神經網絡的四個全連接層，後續還可以做更動。增加層數和神經元數量可以增強模型的表現能力。
// DQN(int input_size, int output_size)：構造函數，用於初始化神經網絡的層數。input_size 是狀態的維度，output_size 是動作的數量。
// forward(torch::Tensor x)：定義了前向傳播的過程，即神經網絡如何從輸入得到輸出。


#ifndef DQN_MODEL_H
#define DQN_MODEL_H

#include <torch/torch.h>

// DQN 模型結構：使用多層全連接神經網絡進行 Q 值預測
struct DQN : torch::nn::Module {
   torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
   torch::nn::Dropout dropout{nullptr};

   // 構造函數：初始化每一層神經網絡
   DQN(int input_size, int output_size);
 
   // 前向傳播函數：定義數據如何在網絡中流動
   torch::Tensor forward(torch::Tensor x);


   // 保存模型權重
   void save_weights(const std::string& file_path);


   // 載入模習權重
   void load_weights(const std::string& file_path);
};

#endif // DQN_MODEL_H
