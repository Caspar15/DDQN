#include "DQNModel.h"

// DQN 類別的構造函數，用來初始化神經網絡的結構。
// input_size: 代表神經網絡輸入的大小，通常與輸入特徵的數量一致（例如，在 5G 模擬中，可能是與狀態特徵相關的數量）。
// output_size: 代表神經網絡輸出的大小，通常與動作空間的大小一致，意味著有多少個可能的動作可以選擇。
DQN::DQN(int input_size, int output_size)
{
    // 增加神經網路的深度和寬度，並使用 Dropout 層，防止過凝合
    fc1 = register_module("fc1", torch::nn::Linear(input_size, 512));
    fc2 = register_module("fc2", torch::nn::Linear(512, 256));
    fc3 = register_module("fc3", torch::nn::Linear(256, 256));
    fc4 = register_module("fc4", torch::nn::Linear(256, 128));
    dropout = register_module("dropout", torch::nn::Dropout(0.5)); // 隨機丟棄 50% 的神經元輸出。

    // Dueling DQN 部分
    state_value = register_module("state_value", torch::nn::Linear(128, 1));                     // 狀態價值
    action_advantage = register_module("action_advantage", torch::nn::Linear(128, output_size)); // 動作優勢
}

// 前向傳播
torch::Tensor DQN::forward(torch::Tensor x)
{
    // 前向傳播，使用 Leaky ReLU 激活函數
    x = torch::leaky_relu(fc1->forward(x), 0.01);
    x = dropout(x); // 在第一層之後套用 Dropout
    x = torch::leaky_relu(fc2->forward(x), 0.01);
    x = torch::leaky_relu(fc3->forward(x), 0.01);
    x = torch::leaky_relu(fc4->forward(x), 0.01);

    // Dueling DQN：計算狀態價值與動作優勢
    torch::Tensor value = state_value->forward(x);
    torch::Tensor advantage = action_advantage->forward(x);

    // 組合狀態價值和動作優勢為 Q 值
    return value + (advantage - advantage.mean());
}

// 保存模型權重
void DQN::save_weights(const std::string &file_path)
{
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(file_path);
}

// 載入模型權重
void DQN::load_weights(const std::string &file_path)
{
    torch::serialize::InputArchive archive;
    archive.load_from(file_path);
    this->load(archive);
}
