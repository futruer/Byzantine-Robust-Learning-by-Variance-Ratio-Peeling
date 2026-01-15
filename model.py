from abc import ABC, abstractmethod
from pydoc import cli
from typing import List
import torch
from torch import nn
import pandas as pd

class Client:

    def __init__(self, model:nn.Module = None, 
                       data:torch.utils.data.DataLoader = None, 
                       reform_func:callable = None,
                       id:int = None):
        self.model = model
        self.data = data
        self.reform_func = reform_func
        self.id = id
        self.is_normal = True

    def get_XY_tlide(self, init_theta:torch.Tensor) -> None:
        
        # 期望labels_features为[label:torch.Tensor, features:torch.Tensor]其中前者一维tensor，后者二维tensor
        labels_features = self.model.extract_features(self.data)
        
        # 期望df的每一行代表一个样本，第一列代表Y_tlide，其余列代表X_tlide的各个特征
        # XY_tlide为pd.DataFrame类型
        XY_tlide = self.reform_func(labels_features, init_theta)
        self.XY_tlide = XY_tlide

    def XY_tlide_least_square(self, init_theta:torch.Tensor) -> torch.Tensor:
        self.get_XY_tlide(init_theta)
        
        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)
        
        # 添加截距项 (全1列)
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)
        
        # 计算最小二乘系数: β = (X^T * X)^(-1) * X^T * Y
        XTX = torch.matmul(X_tlide_with_intercept.T, X_tlide_with_intercept)
        XTY = torch.matmul(X_tlide_with_intercept.T, Y_tlide.unsqueeze(1))
        
        # 使用伪逆来避免奇异矩阵问题
        try:
            coefficients = torch.matmul(torch.linalg.pinv(XTX), XTY)
        except:
            # 如果 pinv 不可用，使用 solve
            coefficients = torch.linalg.solve(XTX, XTY)
        
        return coefficients.squeeze()
    
    def compute_mse_list(self, LS_coefficients: List[dict]) -> None:
        """
        对于传入的每一个最小二乘系数 beta_k，计算当前客户端数据下
        残差向量 Y_tlide - X_tlide^T beta_k 的 L2 范数，并绑定对应的 client id。
        LS_coefficients 是一个字典列表，每个字典包含 {"id": client_id, "beta_k": torch.Tensor}
        """
        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)  # (n,)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)  # (n, d)

        # 添加截距项 (全1列) -> X_with_intercept: (n, d+1)
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)

        # 将所有 beta_k 堆叠成矩阵 B: (d+1, K)，同时记录对应的 client id
        betas = []
        client_ids = []
        for coeff_dict in LS_coefficients:
            beta = coeff_dict["beta_k"]
            client_id = coeff_dict["id"]
            # 确保是一维向量形状 (d+1,)
            if beta.dim() > 1:
                beta = beta.view(-1)
            betas.append(beta)
            client_ids.append(client_id)
        B = torch.stack(betas, dim=1)  # (d+1, K)

        # 预测值矩阵: (n, K)
        preds = X_tlide_with_intercept @ B  # 矩阵乘法

        # 残差矩阵: (n, K)
        residuals = Y_tlide.unsqueeze(1) - preds

        # 对每个 beta_k 计算 L2 范数: 先按样本维度求平方和，再开方 -> (K,)
        l2_norms = torch.sqrt(torch.sum(residuals ** 2, dim=0))

        # 将 L2 范数与对应的 client id 绑定
        self.mse_list = []
        for i, client_id in enumerate(client_ids):
            self.mse_list.append({
                "id": client_id,
                "mse": l2_norms[i]
            })

    def compute_rest_mse_med(self, normal_client_ids:list[int]) -> torch.Tensor:
        """
        从 mse_list 中挑选出 normal_client_ids 对应的元素，
        提取这些元素的 mse 值，计算中位数并返回。
        """
        # 从 mse_list 中筛选出 id 在 normal_client_ids 中的元素
        filtered_mse_values = []
        for mse_item in self.mse_list:
            if mse_item["id"] in normal_client_ids:
                filtered_mse_values.append(mse_item["mse"])
        
        # 如果没有找到任何匹配的元素，返回 None 或抛出异常
        if len(filtered_mse_values) == 0:
            raise ValueError(f"No matching mse values found for normal_client_ids: {normal_client_ids}")
        
        # 将所有 mse 值堆叠成 tensor
        mse_tensor = torch.stack(filtered_mse_values)
        
        # 计算中位数
        mse_med, _ = torch.median(mse_tensor)
        
        return mse_med

    def train(self) -> dict:
        """
        使用当前client的数据进行一次梯度计算。
        返回包含client id和梯度的字典。
        """
        if self.model is None:
            raise ValueError("Model is not set. Please set model before training.")
        if self.data is None:
            raise ValueError("Data is not set. Please set data before training.")
        
        # 设置模型为训练模式
        self.model.train()
        
        # 初始化梯度
        gradients = {}
        
        # 遍历数据计算梯度
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in self.data:
            # 假设batch_data是一个元组 (inputs, labels) 或字典
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                inputs, labels = batch_data
            elif isinstance(batch_data, dict):
                inputs = batch_data.get('input', batch_data.get('x', None))
                labels = batch_data.get('label', batch_data.get('y', None))
            else:
                # 如果batch_data本身就是输入，尝试从model获取标签
                inputs = batch_data
                labels = None
            
            # 前向传播
            if labels is not None:
                outputs = self.model(inputs)
                # 假设模型有loss方法或使用标准损失函数
                if hasattr(self.model, 'loss'):
                    loss = self.model.loss(outputs, labels)
                else:
                    criterion = nn.MSELoss() if outputs.dim() > 1 else nn.MSELoss()
                    loss = criterion(outputs, labels)
            else:
                # 如果没有标签，可能需要使用无监督损失
                outputs = self.model(inputs)
                loss = outputs.mean()  # 简单示例，实际应根据模型调整
            
            # 反向传播计算梯度
            self.model.zero_grad()
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 收集所有参数的梯度
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return {
            "id": self.id,
            "gradients": gradients,
            "loss": total_loss / num_batches if num_batches > 0 else 0.0
        }  

class Orchestrator:
    
    def __init__(self, model:nn.Module, clients:list[Client]):
        self.model = model
        self.clients = clients

        self.init_normal_client_ids()
        self.send_model(self.model)
        
        self.compute_init_theta()
        self.compute_init_client_mse_list()

    def init_normal_client_ids(self) -> None:
        self.normal_client_dict_list: list[dict] = []
        for client in self.clients:
            self.normal_client_dict_list.append({
                "id": client.id,
                "is_normal": client.is_normal
            })

    def send_model(self, model:nn.Module) -> None:
        # 只对 is_normal 为 True 的 client 进行 send_model
        for client in self.clients:
            if client.is_normal:
                client.model = model

    def compute_init_theta(self) -> None:
        self.init_theta:torch.Tensor = self.model.compute_init_theta()

    def collect_client_betas(self) -> None:
        self.LS_coefficients: List[torch.Tensor] = []
        for client in self.clients:
            self.LS_coefficients.append({
                "id": client.id,
                "beta_k": client.XY_tlide_least_square(self.init_theta)
            })
        
    def compute_init_client_mse_list(self) -> None:
        self.collect_client_betas()
        for client in self.clients:
            client.compute_mse_list(self.LS_coefficients)

    def collect_client_mse_med(self) -> list[dict]:
        """
        将 normal_client_ids 发送到每个 Client，
        各个 Client 挑选出正常 client 对应的 mse_list 中的元素并取中位数，
        收集所有 Client 返回的中位数，形成字典列表。
        """
        # 从 normal_client_dict_list 中提取所有 is_normal=True 的 client id
        normal_client_ids = []
        for client_dict in self.normal_client_dict_list:
            if client_dict["is_normal"]:
                normal_client_ids.append(client_dict["id"])
        
        # 将 normal_client_ids 发送到每个 Client，收集返回的中位数
        result_list = []
        for client in self.clients:
            mse_med = client.compute_rest_mse_med(normal_client_ids)
            result_list.append({
                "id": client.id,
                "mse_med": mse_med
            })
        
        return result_list

    def update_normal_clients_by_gap(self, mse_med_list: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据收集到的所有 Client 返回的中位数，对这些中位数进行排序，
        排序后进行一阶差分，找到最大gap，并将最大gap位置及之后所有位置的
        client的is_normal属性改为false，更新normal_client_dict_list。
        返回更新前的方差、更新后的方差以及方差比。
        """
        # 记录更新前 is_normal=True 的 client 的 mse_med 值
        before_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            # 找到对应的 client
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    before_mse_values.append(item["mse_med"])
                    break
        
        # 计算更新前的方差
        if len(before_mse_values) > 1:
            before_mse_tensor = torch.stack(before_mse_values)
            before_variance = torch.var(before_mse_tensor, unbiased=False)
        elif len(before_mse_values) == 1:
            before_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            before_variance = torch.tensor(0.0, dtype=torch.float32)
        
        # 按 mse_med 值进行排序（从小到大）
        sorted_list = sorted(mse_med_list, key=lambda x: x["mse_med"].item())
        
        # 提取排序后的 mse_med 值用于计算差分
        mse_values = [item["mse_med"] for item in sorted_list]
        
        # 计算一阶差分
        if len(mse_values) < 2:
            # 如果只有一个或零个元素，无法计算差分
            # 返回更新前的方差（作为前后方差）和方差比1.0
            return before_variance, before_variance, torch.tensor(1.0, dtype=torch.float32)
        
        # 将 tensor 转换为列表以便计算差分
        mse_values_list = [val.item() for val in mse_values]
        diffs = [mse_values_list[i+1] - mse_values_list[i] for i in range(len(mse_values_list) - 1)]
        
        # 找到最大gap的位置
        max_gap_idx = diffs.index(max(diffs))
        
        # 记录最大gap位置的client id以及之后所有位置的client id
        abnormal_client_ids = []
        # 最大gap在位置i意味着从i到i+1的gap最大，所以应该标记从i+1开始的所有client
        for i in range(max_gap_idx + 1, len(sorted_list)):
            abnormal_client_ids.append(sorted_list[i]["id"])
        
        # 根据这些id将对应client的is_normal属性改为false
        for client in self.clients:
            if client.id in abnormal_client_ids:
                client.is_normal = False
        
        # 更新self.normal_client_dict_list
        self.init_normal_client_ids()
        
        # 记录更新后 is_normal=True 的 client 的 mse_med 值
        after_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            # 找到对应的 client
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    after_mse_values.append(item["mse_med"])
                    break
        
        # 计算更新后的方差
        if len(after_mse_values) > 1:
            after_mse_tensor = torch.stack(after_mse_values)
            after_variance = torch.var(after_mse_tensor, unbiased=False)
        elif len(after_mse_values) == 1:
            after_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            after_variance = torch.tensor(0.0, dtype=torch.float32)
        
        # 计算方差比（后者与前者之比）
        if before_variance.item() != 0:
            variance_ratio = after_variance / before_variance
        else:
            # 如果更新前方差为0，则方差比设为0（或根据业务逻辑设为其他值）
            variance_ratio = torch.tensor(0.0, dtype=torch.float32) if after_variance.item() == 0 else torch.tensor(float('inf'), dtype=torch.float32)
        
        return before_variance, after_variance, variance_ratio

    def _save_client_states(self) -> dict:
        """保存所有client的is_normal状态"""
        states = {}
        for client in self.clients:
            states[client.id] = client.is_normal
        return states
    
    def _restore_client_states(self, states: dict) -> None:
        """恢复所有client的is_normal状态"""
        for client in self.clients:
            if client.id in states:
                client.is_normal = states[client.id]
        self.init_normal_client_ids()

    def iterative_update_normal_clients(self) -> dict:
        """
        迭代更新normal clients，记录最低方差比的时刻，并将该时刻的状态作为最终状态。
        停止条件：当normal client数量小于总client数量的1/2时停止。
        返回包含最终状态信息的字典。
        """
        # 确保从所有client.is_normal均为true开始
        for client in self.clients:
            client.is_normal = True
        self.init_normal_client_ids()
        
        total_clients = len(self.clients)
        min_variance_ratio = float('inf')
        best_state = None
        best_normal_client_dict_list = None
        iteration = 0
        
        # 记录每次迭代的信息
        iteration_records = []
        
        while True:
            # 检查停止条件：normal client数量是否小于总client数量的1/2
            normal_count = sum(1 for client in self.clients if client.is_normal)
            if normal_count < total_clients / 2:
                break
            
            # 收集当前迭代的mse_med
            mse_med_list = self.collect_client_mse_med()
            
            # 如果normal client数量为0，无法继续
            if normal_count == 0:
                break
            
            # 执行更新
            before_var, after_var, variance_ratio = self.update_normal_clients_by_gap(mse_med_list)
            variance_ratio_value = variance_ratio.item()
            
            # 保存更新后的状态
            updated_state = self._save_client_states()
            updated_normal_client_dict_list = [dict(d) for d in self.normal_client_dict_list]
            
            # 记录本次迭代信息
            iteration_records.append({
                "iteration": iteration,
                "variance_ratio": variance_ratio_value,
                "normal_count": normal_count,
                "state": updated_state.copy(),
                "normal_client_dict_list": [dict(d) for d in updated_normal_client_dict_list]
            })
            
            # 检查是否找到更低的方差比（更新后的状态）
            if variance_ratio_value < min_variance_ratio:
                min_variance_ratio = variance_ratio_value
                best_state = updated_state.copy()
                best_normal_client_dict_list = [dict(d) for d in updated_normal_client_dict_list]
            
            iteration += 1
            
            # 检查更新后是否还能继续（防止无限循环）
            new_normal_count = sum(1 for client in self.clients if client.is_normal)
            if new_normal_count == normal_count:
                # 如果没有client被标记为abnormal，说明无法继续更新
                break
        
        # 恢复到最低方差比时刻的状态
        if best_state is not None:
            self._restore_client_states(best_state)
            self.normal_client_dict_list = best_normal_client_dict_list
        
        return {
            "min_variance_ratio": min_variance_ratio,
            "final_normal_client_dict_list": self.normal_client_dict_list,
            "total_iterations": iteration,
            "iteration_records": iteration_records
        }

    def train(self, epochs:int, aggregation:callable) -> None:
        """
        训练流程：
        1. 调用iterative_update_normal_clients确定最终的normal_client_dict_list
        2. 进入epochs次循环，每次循环：
           - 将当前的model发送到每一个正常的client上
           - 通过client.train()收集各个正常client的梯度
           - 将这些梯度打包为列表作为aggregation的传入参数
           - 利用返回值（整合后的梯度）对当前模型的参数进行更新
        """
        # 1. 确定最终的normal_client_dict_list
        self.iterative_update_normal_clients()
        
        # 2. 进入循环梯度更新
        for epoch in range(epochs):
            # 2.1 将当前的model发送到每一个正常的client上
            self.send_model(self.model)
            
            # 2.2 通过client.train()收集各个正常client的梯度
            gradients_list = []
            for client in self.clients:
                if client.is_normal:
                    gradient_result = client.train()
                    gradients_list.append(gradient_result)
            
            # 2.3 将这些梯度打包为列表作为aggregation的传入参数，获取整合后的梯度
            aggregated_gradients = aggregation(gradients_list)
            
            # 2.4 利用整合后的梯度对当前模型的参数进行更新
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in aggregated_gradients:
                        # 假设aggregation返回的是梯度更新量，直接加到参数上
                        # 如果需要使用学习率等，可以在aggregation函数中处理
                        param.data += aggregated_gradients[name]

    
    
        

